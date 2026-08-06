"""Pluggable reader registry for custom file formats."""

from __future__ import annotations

import atexit
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points
import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import TypedDict
from typing import overload

import pooch

from pyvista._warn_external import warn_external
from pyvista.core.utilities._registry_helpers import handler_source
from pyvista.core.utilities.reader import CLASS_READERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista.core.dataset import DataSet


class ReaderHandler(Protocol):
    """Callable that reads *path* and returns a :class:`pyvista.DataSet`."""

    def __call__(self, path: str, /, **kwargs: Any) -> DataSet:
        """Read *path* and return the resulting dataset."""


class ReaderRegistration(NamedTuple):
    """Describe one registered custom reader.

    Returned by :func:`~pyvista.registered_readers`.

    .. versionadded:: 0.48.0

    Attributes
    ----------
    extension : str
        File extension the reader is registered against, including the
        leading dot (e.g. ``'.myformat'``).
    handler : callable
        The reader callable.
    source : str
        Human-readable origin in the form ``'module.qualname'`` for
        explicit registrations or the entry-point ``value`` for
        plugin-discovered registrations.

    """

    extension: str
    handler: ReaderHandler
    source: str


class _RegistryState(TypedDict):
    ext: dict[str, ReaderHandler]
    sources: dict[str, str]
    pending: dict[str, list[EntryPoint]]
    entry_points_loaded: bool


class LocalFileRequiredError(Exception):
    """Raise from a registered reader to signal it needs a local file path.

    When :func:`pyvista.read` passes a remote URI to a custom reader and
    the reader raises this exception, PyVista will download the file to a
    temporary local path and retry the reader automatically.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista.core.utilities.reader_registry import LocalFileRequiredError
    >>> @pv.register_reader('.myformat')  # doctest: +SKIP
    ... def my_reader(path, **kwargs):
    ...     if '://' in path:
    ...         raise LocalFileRequiredError
    ...     ...

    """


_custom_ext_readers: dict[str, ReaderHandler] = {}
_custom_ext_reader_sources: dict[str, str] = {}
# Entry-point metadata, populated by ``_ensure_entry_points``. Maps each
# extension to the list of ``EntryPoint`` records that declared it.
# The plugin module itself is *not* imported until that extension is
# actually requested via :func:`_get_ext_handler`, keeping
# ``pv.read``/``pv.save`` calls for built-in formats free of third-party
# plugin import cost.
_pending_ext_readers: dict[str, list[EntryPoint]] = {}
_entry_points_loaded: bool = False
_temp_files: list[str] = []


def _cleanup_temp_files() -> None:
    """Remove temporary files created by :func:`_download_uri`."""
    for path in _temp_files:
        pathlib.Path(path).unlink(missing_ok=True)
    _temp_files.clear()


atexit.register(_cleanup_temp_files)


def _save_registry_state() -> _RegistryState:
    """Snapshot the current registry state for later restoration."""
    return {
        'ext': _custom_ext_readers.copy(),
        'sources': _custom_ext_reader_sources.copy(),
        'pending': {k: list(v) for k, v in _pending_ext_readers.items()},
        'entry_points_loaded': _entry_points_loaded,
    }


def _restore_registry_state(state: _RegistryState) -> None:
    """Restore registry state from a snapshot."""
    global _entry_points_loaded  # noqa: PLW0603
    _custom_ext_readers.clear()
    _custom_ext_readers.update(state['ext'])
    _custom_ext_reader_sources.clear()
    _custom_ext_reader_sources.update(state['sources'])
    _pending_ext_readers.clear()
    _pending_ext_readers.update({k: list(v) for k, v in state['pending'].items()})
    _entry_points_loaded = state['entry_points_loaded']


def has_scheme(value: str) -> bool:
    """Return ``True`` if *value* starts with a URI scheme (e.g. ``https://``).

    Parameters
    ----------
    value : str
        The string to check.

    Returns
    -------
    bool
        ``True`` if *value* contains a ``://`` scheme prefix before
        the first ``/``.

    """
    # Check that :// appears before the first / to avoid false positives
    # on paths like /data/re://fresh/mesh.vtu
    slash = value.find('/')
    colon = value.find('://')
    return colon > 0 and (slash == -1 or colon < slash)


def _download_uri(uri: str, ext: str) -> str:
    """Download a remote URI to a temporary file, preserving *ext*.

    Uses ``fsspec`` when available (supports ``s3://``, ``gs://``,
    ``az://``, ``http://``, and any other registered filesystem).
    Falls back to ``pooch`` for ``http://`` and ``https://`` URIs.

    Parameters
    ----------
    uri : str
        The remote URI to download.
    ext : str
        File extension to use for the temp file (e.g. ``'.vtu'``).

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
        import fsspec  # noqa: PLC0415  — optional dependency
    except ImportError:
        if not uri.lower().startswith(('http://', 'https://')):
            scheme = uri.split('://', maxsplit=1)[0]
            msg = (
                f'Cannot download "{scheme}://" URIs without fsspec. '
                f'Install it with: pip install fsspec'
            )
            raise ImportError(msg)
        result = pooch.retrieve(uri, known_hash=None, fname=f'pyvista_download{suffix}')  # type: ignore[attr-defined]  # pooch doesn't export retrieve in __all__
        _temp_files.append(result)
        return result

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_name = tmp.name
        _temp_files.append(tmp_name)
        with fsspec.open(uri, 'rb') as remote, pathlib.Path(tmp_name).open('wb') as local:
            shutil.copyfileobj(remote, local)
    except Exception as e:
        msg = f'Failed to download "{uri}": {e}'
        raise ConnectionError(msg) from e
    else:
        return tmp_name


@overload
def register_reader(
    key: str,
    handler: None = None,
    *,
    override: bool = False,
) -> Callable[[ReaderHandler], ReaderHandler]: ...


@overload
def register_reader(
    key: str,
    handler: ReaderHandler,
    *,
    override: bool = False,
) -> None: ...


def register_reader(
    key: str,
    handler: ReaderHandler | None = None,
    *,
    override: bool = False,
) -> Callable[[ReaderHandler], ReaderHandler] | None:
    """Register a custom reader for a file extension.

    Can be used as a plain call or as a decorator.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    key : str
        A file extension (e.g. ``'.myformat'``).

    handler : callable, optional
        A callable with signature ``handler(path: str, **kwargs)`` that
        returns a :class:`pyvista.DataSet`.  When omitted the function
        acts as a decorator and returns the decorated callable unchanged.

    override : bool, default: False
        If ``True``, allow overriding a built-in VTK reader for this
        extension and silence the warning that would otherwise fire when
        replacing an existing custom registration.

    Returns
    -------
    callable or None
        When used as a decorator (``handler`` omitted), returns the
        decorated function.  Otherwise returns ``None``.

    Raises
    ------
    ValueError
        If ``key`` collides with a built-in VTK reader and *override*
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

    Examples
    --------
    Register a reader for a custom file extension.

    >>> import pyvista as pv
    >>> def my_reader(path, **kwargs): ...
    >>> pv.register_reader('.myformat', my_reader)  # doctest: +SKIP

    Use as a decorator.

    >>> @pv.register_reader('.myformat')  # doctest: +SKIP
    ... def my_reader(path, **kwargs): ...

    """
    if handler is None:
        # Decorator form: @pv.register_reader('.ext')
        def _decorator(fn: ReaderHandler) -> ReaderHandler:
            _register(key, fn, override=override)
            return fn

        return _decorator

    _register(key, handler, override=override)
    return None


def _register(
    key: str,
    handler: ReaderHandler,
    *,
    override: bool = False,
    source: str | None = None,
) -> None:
    """Register a handler in the extension registry."""
    key = key.lower()
    if not key.startswith('.'):
        key = f'.{key}'
    if not override and key in CLASS_READERS:
        msg = (
            f'Cannot register custom reader for "{key}": '
            f'collides with built-in VTK reader. '
            f'Use override=True to replace it.'
        )
        raise ValueError(msg)
    if not override and key in _custom_ext_readers:
        existing_source = _custom_ext_reader_sources.get(key, '<unknown>')
        warn_external(
            f'Registering reader for "{key}" replaces an existing custom '
            f'reader from {existing_source}.',
        )
    _custom_ext_readers[key] = handler
    _custom_ext_reader_sources[key] = source if source is not None else handler_source(handler)


def _get_ext_handler(ext: str) -> ReaderHandler | None:
    """Look up a custom extension handler, importing the plugin lazily.

    Built-in extensions never trigger entry-point plugin imports — only
    extensions that an installed plugin has actually claimed do.
    """
    handler = _custom_ext_readers.get(ext)
    if handler is not None:
        return handler
    _ensure_entry_points()
    if ext in _pending_ext_readers:
        _resolve_pending_reader(ext)
    return _custom_ext_readers.get(ext)


def _ensure_entry_points() -> None:
    """Scan ``pyvista.readers`` entry-point metadata once.

    Populates :data:`_pending_ext_readers` with every extension declared
    by an installed plugin. The plugin modules themselves are **not**
    imported here; the cost is one ``importlib.metadata.entry_points``
    call. Plugin imports are deferred to :func:`_resolve_pending_reader`,
    which runs only when a reader for that specific extension is
    actually requested.
    """
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    for ep in entry_points(group='pyvista.readers'):
        key = ep.name.lower()
        if not key.startswith('.'):
            key = f'.{key}'
        if key in _custom_ext_readers:
            continue
        _pending_ext_readers.setdefault(key, []).append(ep)


def _resolve_pending_reader(ext: str) -> bool:
    """Import the plugin claiming *ext*, if any.

    Returns
    -------
    bool
        ``True`` if a plugin loaded successfully for ``ext``. ``False``
        if no pending plugin matches, or if the plugin failed to import.

    Notes
    -----
    A plugin that fails to import emits a ``UserWarning`` and is dropped
    from the pending list, so subsequent lookups of the same extension
    fall straight through without re-triggering the import or
    re-emitting the warning.

    """
    eps = _pending_ext_readers.pop(ext, None)
    if not eps:
        return False
    winner = eps[0]
    try:
        # ep.load() runs third-party import machinery — it can raise
        # literally anything. Convert to a warning so one broken plugin
        # cannot take down every pyvista.read call.
        handler = winner.load()
    except Exception as err:  # noqa: BLE001
        warn_external(
            f'Failed to load pyvista.readers entry point "{winner.value}" for "{ext}": {err}'
        )
        return False
    _custom_ext_readers[ext] = handler
    _custom_ext_reader_sources[ext] = winner.value
    if len(eps) > 1:
        providers = ', '.join(ep.value for ep in eps)
        warn_external(
            f'Multiple pyvista.readers entry points registered for '
            f'"{ext}": {providers}. Using {winner.value}.'
        )
    return True


def _list_custom_exts() -> list[str]:
    """Return the list of extensions with registered custom readers.

    Triggers lazy entry-point *metadata* discovery so that extensions
    contributed by installed packages appear in listings of supported
    formats. The plugin modules themselves are **not** imported.
    """
    _ensure_entry_points()
    return list(_custom_ext_readers.keys() | _pending_ext_readers.keys())


def registered_readers() -> tuple[ReaderRegistration, ...]:
    """Return every custom reader currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every reader visible to PyVista. A plugin that fails to
    import emits a ``UserWarning`` and is skipped; the rest still appear
    in the result.

    .. versionadded:: 0.48.0

    Returns
    -------
    tuple[ReaderRegistration, ...]
        One record per registered extension. Each record exposes
        ``extension``, ``handler``, and ``source``.

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
    _ensure_entry_points()
    for ext in list(_pending_ext_readers):
        _resolve_pending_reader(ext)
    return tuple(
        ReaderRegistration(
            extension=ext,
            handler=handler,
            source=_custom_ext_reader_sources.get(ext, '<unknown>'),
        )
        for ext, handler in _custom_ext_readers.items()
    )
