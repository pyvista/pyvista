"""Pluggable writer registry for custom file formats."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import TypedDict
from typing import overload

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.core.utilities._registry_helpers import handler_source

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista.core.dataobject import DataObject

    class WriterHandler(Protocol):
        """Callable that writes *dataset* to *path*."""

        def __call__(self, dataset: DataObject, path: str, /, **kwargs: Any) -> None:
            """Write *dataset* to *path*, consuming format-specific *kwargs*."""


class WriterRegistration(NamedTuple):
    """Describe one registered custom writer.

    Returned by :func:`~pyvista.registered_writers`.

    .. versionadded:: 0.48.0

    Attributes
    ----------
    extension : str
        File extension the writer is registered against, including the
        leading dot (e.g. ``'.myformat'``).
    handler : callable
        The writer callable.
    source : str
        Human-readable origin in the form ``'module.qualname'`` for
        explicit registrations or the entry-point ``value`` for
        plugin-discovered registrations.

    """

    extension: str
    handler: WriterHandler
    source: str


class _RegistryState(TypedDict):
    ext: dict[str, WriterHandler]
    sources: dict[str, str]
    pending: dict[str, list[EntryPoint]]
    entry_points_loaded: bool


_custom_ext_writers: dict[str, WriterHandler] = {}
_custom_ext_writer_sources: dict[str, str] = {}
# Entry-point metadata, populated by ``_ensure_entry_points``. Maps each
# extension to the list of ``EntryPoint`` records that declared it.
# The plugin module itself is *not* imported until that extension is
# actually requested via :func:`_get_ext_handler`, keeping ``pv.save``
# calls for built-in formats free of third-party plugin import cost.
_pending_ext_writers: dict[str, list[EntryPoint]] = {}
_entry_points_loaded: bool = False
_builtin_writer_exts: frozenset[str] | None = None


def _save_registry_state() -> _RegistryState:
    """Snapshot the current registry state for later restoration."""
    return {
        'ext': _custom_ext_writers.copy(),
        'sources': _custom_ext_writer_sources.copy(),
        'pending': {k: list(v) for k, v in _pending_ext_writers.items()},
        'entry_points_loaded': _entry_points_loaded,
    }


def _restore_registry_state(state: _RegistryState) -> None:
    """Restore registry state from a snapshot."""
    global _entry_points_loaded  # noqa: PLW0603
    _custom_ext_writers.clear()
    _custom_ext_writers.update(state['ext'])
    _custom_ext_writer_sources.clear()
    _custom_ext_writer_sources.update(state['sources'])
    _pending_ext_writers.clear()
    _pending_ext_writers.update({k: list(v) for k, v in state['pending'].items()})
    _entry_points_loaded = state['entry_points_loaded']


def _get_builtin_writer_exts() -> frozenset[str]:
    """Return the union of file extensions handled by built-in PyVista writers."""
    global _builtin_writer_exts  # noqa: PLW0603
    if _builtin_writer_exts is not None:
        return _builtin_writer_exts

    exts: set[str] = set()
    for cls in (
        pv.ImageData,
        pv.RectilinearGrid,
        pv.StructuredGrid,
        pv.PolyData,
        pv.UnstructuredGrid,
        pv.ExplicitStructuredGrid,
        pv.PointSet,
        pv.MultiBlock,
        pv.PartitionedDataSet,
    ):
        writers = getattr(cls, '_WRITERS', None) or {}
        exts.update(writers.keys())

    _builtin_writer_exts = frozenset(exts)
    return _builtin_writer_exts


@overload
def register_writer(
    key: str,
    handler: None = None,
    *,
    override: bool = False,
) -> Callable[[WriterHandler], WriterHandler]: ...


@overload
def register_writer(
    key: str,
    handler: WriterHandler,
    *,
    override: bool = False,
) -> None: ...


def register_writer(
    key: str,
    handler: WriterHandler | None = None,
    *,
    override: bool = False,
) -> Callable[[WriterHandler], WriterHandler] | None:
    """Register a custom writer for a file extension.

    Can be used as a plain call or as a decorator.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    key : str
        A file extension (e.g. ``'.myformat'``).

    handler : callable, optional
        A callable with signature ``handler(dataset, path, **kwargs)`` that
        writes *dataset* to *path*.  Any extra keyword arguments passed to
        :meth:`pyvista.DataObject.save` are forwarded to the handler as
        ``**kwargs`` — use them to expose format-specific options such as
        compression level, thread count, or chunking.  Handlers that do
        not need per-call options can omit ``**kwargs``; a call to
        :meth:`~pyvista.DataObject.save` that passes extras to such a
        handler will raise :class:`TypeError` from Python itself.  When
        ``handler`` is omitted the function acts as a decorator and
        returns the decorated callable unchanged.

    override : bool, default: False
        If ``True``, allow overriding a built-in PyVista writer for this
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
        If ``key`` collides with a built-in PyVista writer and *override*
        is ``False``.

    Warns
    -----
    UserWarning
        If ``key`` already refers to a registered custom writer. The new
        registration replaces the old one (last wins); pass
        ``override=True`` to silence the warning.

    See Also
    --------
    pyvista.register_reader
        Sibling API for registering custom readers.
    pyvista.registered_writers
        Introspect every registered writer.

    Notes
    -----
    When :meth:`pyvista.DataObject.save` is called, registered custom
    writers are dispatched *before* built-in VTK writers — mirroring
    the dispatch order of :func:`pyvista.read`.  Passing
    ``override=True`` is therefore the only way to replace a built-in
    writer at save time.

    Any keyword arguments passed to :meth:`~pyvista.DataObject.save`
    beyond its documented parameters are forwarded verbatim to the
    registered handler.  When no custom writer is registered for the
    target extension, extra keyword arguments raise :class:`TypeError`
    from :meth:`~pyvista.DataObject.save` — PyVista never silently
    drops writer options.

    Examples
    --------
    Register a writer for a custom file extension with a
    format-specific option.

    >>> import pyvista as pv
    >>> def my_writer(dataset, path, *, level=3): ...
    >>> pv.register_writer('.myformat', my_writer)  # doctest:+SKIP
    >>> pv.Sphere().save('sphere.myformat', level=9)  # doctest:+SKIP

    Use as a decorator.

    >>> @pv.register_writer('.myformat')  # doctest:+SKIP
    ... def my_writer(dataset, path, **kwargs): ...

    """
    if handler is None:
        # Decorator form: @pv.register_writer('.ext')
        def _decorator(fn: WriterHandler) -> WriterHandler:
            _register(key, fn, override=override)
            return fn

        return _decorator

    _register(key, handler, override=override)
    return None


def _register(
    key: str,
    handler: WriterHandler,
    *,
    override: bool = False,
    source: str | None = None,
) -> None:
    """Register a handler in the extension registry."""
    key = key.lower()
    if not key.startswith('.'):
        key = f'.{key}'
    if not override and key in _get_builtin_writer_exts():
        msg = (
            f'Cannot register custom writer for "{key}": '
            f'collides with a built-in PyVista writer. '
            f'Use override=True to replace it.'
        )
        raise ValueError(msg)
    if not override and key in _custom_ext_writers:
        existing_source = _custom_ext_writer_sources.get(key, '<unknown>')
        warn_external(
            f'Registering writer for "{key}" replaces an existing custom '
            f'writer from {existing_source}.',
        )
    _custom_ext_writers[key] = handler
    _custom_ext_writer_sources[key] = source if source is not None else handler_source(handler)


def _get_ext_handler(ext: str) -> WriterHandler | None:
    """Look up a custom extension handler, importing the plugin lazily.

    Built-in extensions never trigger entry-point plugin imports — only
    extensions that an installed plugin has actually claimed do.
    """
    handler = _custom_ext_writers.get(ext)
    if handler is not None:
        return handler
    _ensure_entry_points()
    if ext in _pending_ext_writers:
        _resolve_pending_writer(ext)
    return _custom_ext_writers.get(ext)


def _ensure_entry_points() -> None:
    """Scan ``pyvista.writers`` entry-point metadata once.

    Populates :data:`_pending_ext_writers` with every extension declared
    by an installed plugin. The plugin modules themselves are **not**
    imported here; the cost is one ``importlib.metadata.entry_points``
    call. Plugin imports are deferred to :func:`_resolve_pending_writer`,
    which runs only when a writer for that specific extension is
    actually requested.
    """
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    for ep in entry_points(group='pyvista.writers'):
        key = ep.name.lower()
        if not key.startswith('.'):
            key = f'.{key}'
        if key in _custom_ext_writers:
            continue
        _pending_ext_writers.setdefault(key, []).append(ep)


def _resolve_pending_writer(ext: str) -> bool:
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
    eps = _pending_ext_writers.pop(ext, None)
    if not eps:
        return False
    winner = eps[0]
    try:
        # ep.load() runs third-party import machinery — it can raise
        # literally anything. Convert to a warning so one broken plugin
        # cannot take down every pyvista.save call.
        handler = winner.load()
    except Exception as err:  # noqa: BLE001
        warn_external(
            f'Failed to load pyvista.writers entry point "{winner.value}" for "{ext}": {err}'
        )
        return False
    _custom_ext_writers[ext] = handler
    _custom_ext_writer_sources[ext] = winner.value
    if len(eps) > 1:
        providers = ', '.join(ep.value for ep in eps)
        warn_external(
            f'Multiple pyvista.writers entry points registered for '
            f'"{ext}": {providers}. Using {winner.value}.'
        )
    return True


def _list_custom_exts() -> list[str]:
    """Return the list of extensions with registered custom writers.

    Triggers lazy entry-point *metadata* discovery so that extensions
    contributed by installed packages appear in error messages listing
    supported formats. The plugin modules themselves are **not**
    imported.
    """
    _ensure_entry_points()
    return list(_custom_ext_writers.keys() | _pending_ext_writers.keys())


def registered_writers() -> tuple[WriterRegistration, ...]:
    """Return every custom writer currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every writer visible to PyVista. A plugin that fails to
    import emits a ``UserWarning`` and is skipped; the rest still appear
    in the result.

    .. versionadded:: 0.48.0

    Returns
    -------
    tuple[WriterRegistration, ...]
        One record per registered extension. Each record exposes
        ``extension``, ``handler``, and ``source``.

    Examples
    --------
    >>> import pyvista as pv
    >>> def my_writer(dataset, path, **kwargs): ...
    >>> pv.register_writer('.demo_writer', my_writer)
    >>> [
    ...     r.extension
    ...     for r in pv.registered_writers()
    ...     if r.extension == '.demo_writer'
    ... ]
    ['.demo_writer']

    """
    _ensure_entry_points()
    for ext in list(_pending_ext_writers):
        _resolve_pending_writer(ext)
    return tuple(
        WriterRegistration(
            extension=ext,
            handler=handler,
            source=_custom_ext_writer_sources.get(ext, '<unknown>'),
        )
        for ext, handler in _custom_ext_writers.items()
    )
