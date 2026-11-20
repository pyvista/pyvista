"""Pluggable reader registry for custom file formats."""

from __future__ import annotations

import atexit
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import TypedDict
from typing import TypeVar
from typing import Union
from typing import cast
from typing import overload

import pooch

from pyvista.core.utilities._optional_formats import _READ
from pyvista.core.utilities._optional_formats import _declared_reader_class
from pyvista.core.utilities._optional_formats import _import_handler
from pyvista.core.utilities._optional_formats import _installed_extensions
from pyvista.core.utilities._optional_formats import _missing_message
from pyvista.core.utilities._optional_formats import _source
from pyvista.core.utilities._registry import EntryPointRefusedError
from pyvista.core.utilities._registry import _HandlerRegistry
from pyvista.core.utilities._registry import _HandlerRegistryState
from pyvista.core.utilities.reader import _CLASS_READERS
from pyvista.core.utilities.reader import BaseReader

if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.metadata import EntryPoint

    from pyvista.core.dataset import DataSet


class ReaderHandler(Protocol):
    """Callable that reads ``path`` and returns a :class:`pyvista.DataSet`."""

    def __call__(self, path: str, /, **kwargs: Any) -> DataSet:
        """Read ``path`` and return the resulting dataset."""


# A bare callable, or a BaseReader subclass.
ReaderProvider = Union[ReaderHandler, type['BaseReader[Any]']]  # noqa: UP007


def _is_reader_class(obj: object) -> bool:
    """Return ``True`` when *obj* is a :class:`~pyvista.BaseReader` subclass."""
    return isinstance(obj, type) and issubclass(obj, BaseReader)


READER_GROUP = 'pyvista.readers'

# Declaring the intent in metadata is what separates a deliberate replacement
# from an accidental one, so this group is silent and the other is an error.
READER_OVERRIDE_GROUP = 'pyvista.readers.override'


class ReaderRegistration(NamedTuple):
    """Describe one registered custom reader.

    Returned by :func:`~pyvista.registered_readers`.

    .. versionadded:: 0.48.0

    Attributes
    ----------
    extension : str
        File extension the reader is registered against, including the
        leading dot (for example, ``'.myformat'``).
    handler : callable or type[pyvista.BaseReader]
        The reader callable, or the :class:`~pyvista.BaseReader`
        subclass when the extension was registered with a reader class.
    source : str
        Human-readable origin in the form ``'module.qualname'`` for
        explicit registrations or the entry-point ``value`` for
        plugin-discovered registrations.
    reader_class : bool
        ``True`` when ``handler`` is a :class:`~pyvista.BaseReader`
        subclass, meaning :func:`pyvista.get_reader` resolves this
        extension as well as :func:`pyvista.read`.

        .. versionadded:: 0.49.0

    override : bool
        ``True`` when this reader replaces one PyVista ships, either
        through ``register_reader(..., override=True)`` or through the
        ``pyvista.readers.override`` entry-point group. Check this first
        when a built-in format reads differently than expected: it names
        the plugin that took the extension over.

        .. versionadded:: 0.49.0

    """

    extension: str
    handler: ReaderProvider
    source: str
    reader_class: bool = False
    override: bool = False


class _RegistryState(TypedDict):
    """Mutable state of the reader registry."""

    registry: _HandlerRegistryState
    overrides: set[str]


class LocalFileRequiredError(Exception):
    """Raise from a registered reader to signal it needs a local file path.

    When :func:`pyvista.read` passes a remote URI to a custom reader and
    the reader raises this exception, PyVista will download the file to a
    temporary local path and retry the reader automatically.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista.core.utilities.reader_registry import LocalFileRequiredError
    >>> @pv.register_reader('.myremoteformat')  # doctest: +SKIP
    ... def my_reader(path, **kwargs):
    ...     if '://' in path:
    ...         raise LocalFileRequiredError
    ...     ...

    """


def _normalize_ext(key: str) -> str:
    """Return ``key`` in lowercase, with a leading dot."""
    key = key.lower()
    return key if key.startswith('.') else f'.{key}'


class _ReaderRegistry(_HandlerRegistry[ReaderProvider]):
    """Reader registry: the override group may replace a built-in, the plain group may not."""

    def pending_key(self, ep: EntryPoint, group: str) -> str | None:  # numpydoc ignore=PR01
        """Record an override-group entry point as a deliberate replacement."""
        key = super().pending_key(ep, group)
        if key is not None and group == READER_OVERRIDE_GROUP:
            _override_ext_readers.add(key)
        return key

    def load(self, key: str, eps: list[EntryPoint]) -> None:  # numpydoc ignore=PR01
        """Refuse a plain-group entry point that would shadow a built-in reader."""
        if key in _CLASS_READERS and key not in _override_ext_readers:
            raise EntryPointRefusedError(_undeclared_override_message(key, eps[0]))
        super().load(key, eps)


# One table for callables and reader classes: ``_get_ext_handler`` serves the
# first, ``_get_ext_reader_class`` the second. The ordinary group is scanned
# first, so an entry point declared in both resolves to the strict reading.
_registry = _ReaderRegistry(
    READER_GROUP,
    READER_OVERRIDE_GROUP,
    label='reader for',
    normalize=_normalize_ext,
    builtin=lambda ext: ext in _CLASS_READERS,
    builtin_label='built-in VTK reader',
)
_custom_readers = _registry.explicit
_plugins = _registry.plugins
# Extensions whose custom reader replaces a built-in on purpose.
_override_ext_readers: set[str] = set()


def _custom_class_readers() -> dict[str, type[BaseReader[Any]]]:
    """Return the registered reader classes by extension."""
    return {
        ext: cast('type[BaseReader[Any]]', handler)
        for ext, handler in _custom_readers.items()
        if _is_reader_class(handler)
    }


_temp_files: list[str] = []
_temp_dirs: list[str] = []


def _cleanup_temp_files() -> None:
    """Remove temporary files and directories created by :func:`_download_uri`."""
    for path in _temp_files:
        Path(path).unlink(missing_ok=True)
    _temp_files.clear()
    for path in _temp_dirs:
        shutil.rmtree(path, ignore_errors=True)
    _temp_dirs.clear()


atexit.register(_cleanup_temp_files)


def _save_registry_state() -> _RegistryState:
    """Snapshot the current registry state for later restoration."""
    return {'registry': _registry.snapshot(), 'overrides': _override_ext_readers.copy()}


def _restore_registry_state(state: _RegistryState) -> None:
    """Restore registry state from a snapshot."""
    _registry.restore(state['registry'])
    _override_ext_readers.clear()
    _override_ext_readers.update(state['overrides'])


def has_scheme(value: str) -> bool:
    """Return ``True`` if ``value`` starts with a URI scheme (for example, ``https://``).

    Parameters
    ----------
    value : str
        The string to check.

    Returns
    -------
    bool
        ``True`` if ``value`` contains a ``://`` scheme prefix before
        the first ``/``.

    """
    # Check that :// appears before the first / to avoid false positives
    # on paths like /data/re://fresh/mesh.vtu
    slash = value.find('/')
    colon = value.find('://')
    return colon > 0 and (slash == -1 or colon < slash)


def _download_uri(uri: str, ext: str) -> str:
    """Download a remote URI to a temporary file, preserving ``ext``.

    Uses ``fsspec`` when available (supports ``s3://``, ``gs://``,
    ``az://``, ``http://``, and any other registered filesystem).
    Falls back to ``pooch`` for ``http://`` and ``https://`` URIs.

    Parameters
    ----------
    uri : str
        The remote URI to download.
    ext : str
        File extension to use for the temp file (for example, ``'.vtu'``).

    Returns
    -------
    str
        Path to the downloaded temporary file.

    Raises
    ------
    ImportError
        If the URI scheme requires ``fsspec`` and it is not installed.
    ConnectionError
        If the download fails.

    """
    suffix = ext or ''
    try:
        import fsspec  # noqa: PLC0415 -- optional dependency
    except ImportError:
        if not uri.lower().startswith(('http://', 'https://')):
            scheme = uri.split('://', maxsplit=1)[0]
            msg = (
                f'Cannot download "{scheme}://" URIs without fsspec. '
                f'Install it with: pip install fsspec'
            )
            raise ImportError(msg)
        # A fresh directory per call: pooch reuses an existing file of the same name
        # without checking it came from the same URL.
        download_dir = tempfile.mkdtemp(prefix='pyvista_download_')
        _temp_dirs.append(download_dir)
        result = pooch.retrieve(  # type: ignore[attr-defined]  # pooch doesn't export retrieve in __all__
            uri, known_hash=None, fname=f'download{suffix}', path=download_dir
        )
        _temp_files.append(result)
        return result

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_name = tmp.name
        _temp_files.append(tmp_name)
        with fsspec.open(uri, 'rb') as remote, Path(tmp_name).open('wb') as local:
            shutil.copyfileobj(remote, local)
    except Exception as e:
        msg = f'Failed to download "{uri}": {e}'
        raise ConnectionError(msg) from e
    else:
        return tmp_name


_T_Provider = TypeVar('_T_Provider', bound=ReaderProvider)


# fmt: off
# ruff: disable[E501]
@overload
def register_reader(key: str, handler: None = None, *, override: bool = False) -> Callable[[_T_Provider], _T_Provider]: ...
@overload
def register_reader(key: str, handler: ReaderProvider, *, override: bool = False) -> None: ...
# ruff: enable[E501]
# fmt: on


def register_reader(
    key: str,
    handler: ReaderProvider | None = None,
    *,
    override: bool = False,
) -> Callable[[_T_Provider], _T_Provider] | None:
    """Register a custom reader for a file extension.

    Can be used as a plain call or as a decorator.

    Two kinds of reader may be registered:

    * A :class:`~pyvista.BaseReader` **subclass**. This is the preferred
      form. The class is resolved by :func:`pyvista.get_reader` exactly
      like a built-in reader, so keyword arguments passed to
      :func:`pyvista.read` set reader attributes, ``progress_bar`` and
      ``validate`` are honored, VTK error observers are attached, and
      mixins such as :class:`~pyvista.TimeReader` and
      :class:`~pyvista.PointCellDataSelection` work unchanged.

    * A bare **callable** ``handler(path, **kwargs)``. This is the
      lighter form for a format that has no reader-level state to
      expose. :func:`pyvista.read` calls it directly, forwarding its
      ``**kwargs``; :func:`pyvista.get_reader` raises
      :class:`ValueError` for the extension because there is no reader
      object to hand back. A callable registered with ``override=True``
      is the exception: reader arguments for an extension PyVista
      already reads route to the built-in reader instead.

    .. versionadded:: 0.48.0

    .. versionchanged:: 0.49.0
        ``handler`` may be a :class:`~pyvista.BaseReader` subclass.

    Parameters
    ----------
    key : str
        A file extension (for example, ``'.myformat'``).

    handler : callable or type[pyvista.BaseReader], optional
        A :class:`~pyvista.BaseReader` subclass, or a callable with
        signature ``handler(path: str, **kwargs)`` that returns a
        :class:`pyvista.DataSet`. When omitted the function acts as a
        decorator and returns the decorated object unchanged.

    override : bool, default: False
        If ``True``, allow overriding a built-in VTK reader for this
        extension and silence the warning that would otherwise fire when
        replacing an existing custom registration. The equivalent for a
        plugin discovered through entry points is to declare the entry
        point in the ``pyvista.readers.override`` group rather than
        ``pyvista.readers``.

    Returns
    -------
    callable or None
        When used as a decorator (``handler`` omitted), returns the
        decorated function or class. Otherwise returns ``None``.

    Raises
    ------
    ValueError
        If ``key`` collides with a built-in VTK reader and ``override``
        is ``False``.

    Warns
    -----
    UserWarning
        If ``key`` already refers to a registered custom reader. The new
        registration replaces the old one (last wins); pass
        ``override=True`` to silence the warning.

    See Also
    --------
    pyvista.register_writer
        Sibling API for registering custom writers.
    pyvista.registered_readers
        Introspect every registered reader.
    pyvista.get_reader
        Resolves registered :class:`~pyvista.BaseReader` subclasses.

    Examples
    --------
    Register a reader for a custom file extension.

    >>> import pyvista as pv
    >>> def my_reader(path, **kwargs): ...
    >>> pv.register_reader('.myformat', my_reader)  # doctest: +SKIP

    Use as a decorator.

    >>> @pv.register_reader('.myotherformat')  # doctest: +SKIP
    ... def my_reader(path, **kwargs): ...

    Register a :class:`~pyvista.BaseReader` subclass so that
    :func:`pyvista.get_reader` resolves the extension too. Where VTK has no
    reader for the format, :class:`~pyvista.BaseVTKReader` supplies the
    parsing half.

    >>> class _MyVTKReader(pv.BaseVTKReader):
    ...     def UpdateInformation(self): ...
    ...     def Update(self):
    ...         self._data_object = pv.PolyData()
    >>> @pv.register_reader('.mybinaryformat')  # doctest: +SKIP
    ... class MyReader(pv.BaseReader):
    ...     _class_reader = _MyVTKReader

    """
    if handler is None:
        # Decorator form: @pv.register_reader('.ext')
        def _decorator(obj: _T_Provider) -> _T_Provider:
            _register(key, obj, override=override)
            return obj

        return _decorator

    _register(key, handler, override=override)
    return None


def _register(
    key: str,
    handler: ReaderProvider,
    *,
    override: bool = False,
    source: str | None = None,
) -> None:
    """Register a handler or reader class in the extension registry."""
    key = _registry.register(key, handler, source, override=override)
    if override:
        _override_ext_readers.add(key)
    else:
        _override_ext_readers.discard(key)


def _get_ext_handler(ext: str) -> ReaderHandler | None:
    """Look up a custom extension *callable*, importing the plugin lazily.

    Returns ``None`` for an extension registered with a
    :class:`~pyvista.BaseReader` subclass: those are resolved by
    :func:`pyvista.get_reader` instead, so that :func:`pyvista.read`
    falls through to the same code path built-in readers use.

    Built-in extensions never trigger entry-point plugin imports—only
    extensions that an installed plugin has actually claimed do.
    """
    _resolve_ext(ext)
    handler = _custom_readers.get(ext)
    return None if handler is None or _is_reader_class(handler) else cast('ReaderHandler', handler)


def _get_ext_reader_class(ext: str) -> type[BaseReader[Any]] | None:
    """Look up a registered reader class, importing the plugin lazily.

    Consulted by :func:`pyvista.get_reader` ahead of ``_CLASS_READERS``, so an
    override wins on that path as well as on :func:`pyvista.read`.
    """
    ext = _normalize_ext(ext)
    _resolve_ext(ext)
    handler = _custom_readers.get(ext)
    return cast('type[BaseReader[Any]]', handler) if _is_reader_class(handler) else None


def _missing_reader_message(ext: str, filename: str | None = None) -> str | None:
    """Return install instructions when ``ext`` needs a companion package PyVista cannot import."""
    return _missing_message(ext, _READ, filename)


def _optional_reader_class_name(ext: str) -> str | None:
    """Return the reader class an optional format exposes, when it is importable."""
    handler, _ = _import_handler(ext, _READ)
    return _declared_reader_class(ext) if handler is not None else None


def _resolve_optional_reader(ext: str) -> bool:
    """Register the companion package serving ``ext``, when it is importable."""
    handler, _ = _import_handler(ext, _READ)
    if handler is None:
        return False
    _register(ext, cast('ReaderHandler', handler), source=_source(ext, _READ))
    return True


def _resolve_ext(ext: str) -> None:
    """Make sure any plugin claiming ``ext`` has been imported."""
    if ext in _custom_readers:
        return
    if _plugins.resolve(ext):
        return
    _resolve_optional_reader(ext)


def _entry_point_package(ep: EntryPoint) -> str:
    """Return the distribution name behind ``ep``, best effort.

    ``EntryPoint.dist`` is unset for one constructed by hand, so fall back to
    the module half of its value.
    """
    dist = getattr(ep, 'dist', None)
    name = getattr(dist, 'name', None)
    if isinstance(name, str) and name:
        return name
    return str(ep.value).partition(':')[0]


def _undeclared_override_message(ext: str, ep: EntryPoint) -> str:
    """Build the error raised when a plugin shadows a built-in by accident."""
    builtin = _CLASS_READERS[ext].__name__
    return (
        f'The package "{_entry_point_package(ep)}" declares a "{ext}" reader in the '
        f'"{READER_GROUP}" entry-point group, but PyVista already reads "{ext}" with '
        f'{builtin}. Registering it would silently change what every '
        f'`pyvista.read("...{ext}")` call in this environment returns, so PyVista '
        f'refuses rather than guess.\n'
        f'If replacing {builtin} is intended, the package should declare the entry '
        f'point in the "{READER_OVERRIDE_GROUP}" group instead:\n\n'
        f'    [project.entry-points."{READER_OVERRIDE_GROUP}"]\n'
        f'    "{ext}" = "{ep.value}"\n\n'
        f'If it is not intended, the package should claim a different extension. '
        f'Uninstalling "{_entry_point_package(ep)}" also restores {builtin}.'
    )


def _list_custom_exts() -> list[str]:
    """Return the list of extensions with registered custom readers.

    Triggers lazy entry-point *metadata* discovery so that extensions
    contributed by installed packages appear in listings of supported
    formats. The plugin modules themselves are **not** imported.
    """
    return list(_registry.keys() | _installed_extensions(_READ))


def registered_readers() -> tuple[ReaderRegistration, ...]:
    """Return every custom reader currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every reader visible to PyVista. A plugin that fails to
    import emits a ``UserWarning`` and is skipped; the rest still appear
    in the result.

    This is the call to reach for when a built-in format reads
    differently than expected: a record whose ``override`` is ``True``
    names the plugin that took the extension over, and its ``source``
    says where that plugin came from.

    .. versionadded:: 0.48.0

    .. versionchanged:: 0.49.0
        Records carry ``reader_class`` and ``override``.

    .. versionchanged:: 0.50.0
        A plugin whose import failed stays pending and is retried on
        every call, so a reader whose dependency was installed after the
        failure becomes available again.

    Returns
    -------
    tuple[ReaderRegistration, ...]
        One record per registered extension. Each record exposes
        ``extension``, ``handler``, ``source``, ``reader_class``, and
        ``override``.

    Examples
    --------
    >>> import pyvista as pv
    >>> def my_reader(path, **kwargs): ...
    >>> pv.register_reader('.demo_reader', my_reader)
    >>> [
    ...     r.extension
    ...     for r in pv.registered_readers()
    ...     if r.extension == '.demo_reader'
    ... ]
    ['.demo_reader']

    """
    return tuple(
        ReaderRegistration(
            extension=ext,
            handler=handler,
            source=source,
            reader_class=_is_reader_class(handler),
            override=ext in _override_ext_readers,
        )
        for ext, handler, source in _registry.registered()
    )
