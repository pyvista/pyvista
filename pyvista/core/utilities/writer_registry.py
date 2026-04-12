"""Pluggable writer registry for custom file formats."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import TypedDict
from typing import overload

import pyvista as pv
from pyvista._warn_external import warn_external

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista.core.dataobject import DataObject

    class WriterHandler(Protocol):
        """Callable that writes *dataset* to *path*."""

        def __call__(self, dataset: DataObject, path: str, /, **kwargs: Any) -> None:
            """Write *dataset* to *path*, consuming format-specific *kwargs*."""


class _RegistryState(TypedDict):
    ext: dict[str, WriterHandler]
    entry_points_loaded: bool


_custom_ext_writers: dict[str, WriterHandler] = {}
_entry_points_loaded: bool = False
_builtin_writer_exts: frozenset[str] | None = None


def _save_registry_state() -> _RegistryState:
    """Snapshot the current registry state for later restoration."""
    return {
        'ext': _custom_ext_writers.copy(),
        'entry_points_loaded': _entry_points_loaded,
    }


def _restore_registry_state(state: _RegistryState) -> None:
    """Restore registry state from a snapshot."""
    global _entry_points_loaded  # noqa: PLW0603
    _custom_ext_writers.clear()
    _custom_ext_writers.update(state['ext'])
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
        extension.  When ``False`` (the default), registering a handler
        for an extension that collides with a built-in writer raises
        :class:`ValueError`.

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

    See Also
    --------
    pyvista.register_reader
        Sibling API for registering custom readers.

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

    .. code-block:: python

        import pyvista as pv


        def my_writer(dataset, path, *, level=3): ...


        pv.register_writer('.myformat', my_writer)
        pv.Sphere().save('sphere.myformat', level=9)

    Use as a decorator.

    .. code-block:: python

        @pv.register_writer('.myformat')
        def my_writer(dataset, path, **kwargs): ...

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
    _custom_ext_writers[key] = handler


def _get_ext_handler(ext: str) -> WriterHandler | None:
    """Look up a custom extension handler, discovering entry points lazily."""
    handler = _custom_ext_writers.get(ext)
    if handler is not None:
        return handler
    _ensure_entry_points()
    return _custom_ext_writers.get(ext)


def _ensure_entry_points() -> None:
    """Scan the ``pyvista.writers`` entry-point group and load handlers."""
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    # Group entry points by normalized key so we can detect duplicate
    # providers *and* guarantee a deterministic winner (the first one).
    eps_by_key: dict[str, list[EntryPoint]] = {}
    for ep in entry_points(group='pyvista.writers'):
        key = ep.name.lower()
        if not key.startswith('.'):
            key = f'.{key}'
        eps_by_key.setdefault(key, []).append(ep)

    for key, eps in eps_by_key.items():
        if key in _custom_ext_writers:
            continue
        winner = eps[0]
        try:
            # ep.load() runs third-party import machinery — it can raise
            # literally anything. Convert to a warning so one broken plugin
            # cannot take down every pyvista.save call.
            _custom_ext_writers[key] = winner.load()
        except Exception as err:  # noqa: BLE001
            warn_external(
                f'Failed to load pyvista.writers entry point "{winner.value}" for "{key}": {err}'
            )
            continue
        if len(eps) > 1:
            providers = ', '.join(ep.value for ep in eps)
            warn_external(
                f'Multiple pyvista.writers entry points registered for '
                f'"{key}": {providers}. Using {winner.value}.'
            )


def _list_custom_exts() -> list[str]:
    """Return the list of extensions with registered custom writers.

    Triggers lazy entry-point discovery so that extensions contributed
    by installed packages appear in error messages listing supported
    formats.
    """
    _ensure_entry_points()
    return list(_custom_ext_writers.keys())
