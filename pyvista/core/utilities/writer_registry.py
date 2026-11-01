"""Pluggable writer registry for custom file formats."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import overload

import pyvista as pv
from pyvista.core.utilities._optional_formats import _WRITE
from pyvista.core.utilities._optional_formats import _missing_message
from pyvista.core.utilities._registry import _HandlerRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista.core.dataobject import DataObject


class WriterHandler(Protocol):
    """Callable that writes ``dataset`` to ``path``."""

    def __call__(self, dataset: DataObject, path: str, /, **kwargs: Any) -> None:
        """Write ``dataset`` to ``path``, consuming format-specific ``kwargs``."""


class WriterRegistration(NamedTuple):
    """Describe one registered custom writer.

    Returned by :func:`~pyvista.registered_writers`.

    .. versionadded:: 0.48.0

    Attributes
    ----------
    extension : str
        File extension the writer is registered against, including the
        leading dot (for example, ``'.myformat'``).
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


WRITER_GROUP = 'pyvista.writers'


def _normalize_ext(key: str) -> str:
    """Return ``key`` in lowercase, with a leading dot."""
    key = key.lower()
    return key if key.startswith('.') else f'.{key}'


_registry: _HandlerRegistry[WriterHandler] = _HandlerRegistry(
    WRITER_GROUP,
    label='writer for',
    normalize=_normalize_ext,
    builtin=lambda ext: ext in _get_builtin_writer_exts(),
    builtin_label='a built-in PyVista writer',
)
_custom_ext_writers = _registry.explicit
_plugins = _registry.plugins
_save_registry_state = _registry.snapshot
_restore_registry_state = _registry.restore
_builtin_writer_exts: frozenset[str] | None = None


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


# fmt: off
# ruff: disable[E501]
@overload
def register_writer(key: str, handler: None = None, *, override: bool = False) -> Callable[[WriterHandler], WriterHandler]: ...
@overload
def register_writer(key: str, handler: WriterHandler, *, override: bool = False) -> None: ...
# ruff: enable[E501]
# fmt: on


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
        A file extension (for example, ``'.myformat'``).

    handler : callable, optional
        A callable with signature ``handler(dataset, path, **kwargs)`` that
        writes ``dataset`` to ``path``. Any extra keyword arguments passed to
        :meth:`pyvista.DataObject.save` are forwarded to the handler as
        ``**kwargs``—use them to expose format-specific options such as
        compression level, thread count, or chunking. Handlers that do
        not need per-call options can omit ``**kwargs``; a call to
        :meth:`~pyvista.DataObject.save` that passes extras to such a
        handler will raise :class:`TypeError` from Python itself. When
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
        decorated function. Otherwise returns ``None``.

    Raises
    ------
    ValueError
        If ``key`` collides with a built-in PyVista writer and ``override``
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
    writers are dispatched *before* built-in VTK writers—mirroring
    the dispatch order of :func:`pyvista.read`. Passing
    ``override=True`` is therefore the only way to replace a built-in
    writer at save time.

    Any keyword arguments passed to :meth:`~pyvista.DataObject.save`
    beyond its documented parameters are forwarded verbatim to the
    registered handler. When no custom writer is registered for the
    target extension, extra keyword arguments raise :class:`TypeError`
    from :meth:`~pyvista.DataObject.save`—PyVista never silently
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
            _registry.register(key, fn, override=override)
            return fn

        return _decorator

    _registry.register(key, handler, override=override)
    return None


def _missing_writer_message(ext: str, filename: str | None = None) -> str | None:
    """Return install instructions when ``ext`` needs a companion package PyVista cannot import."""
    return _missing_message(ext, _WRITE, filename)


def _get_ext_handler(ext: str) -> WriterHandler | None:
    """Look up a custom extension handler, importing its plugin on first request."""
    return _registry.get(ext)


def _list_custom_exts() -> list[str]:
    """Return the extensions with a custom writer, registered or still pending."""
    return list(_registry.keys())


def registered_writers() -> tuple[WriterRegistration, ...]:
    """Return every custom writer currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every writer visible to PyVista. A plugin that fails to
    import emits a ``UserWarning`` and is skipped; the rest still appear
    in the result.

    .. versionadded:: 0.48.0

    .. versionchanged:: 0.50.0
        A plugin whose import failed stays pending and is retried on
        every call, so a writer whose dependency was installed after the
        failure becomes available again.

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
    return tuple(WriterRegistration(*record) for record in _registry.registered())
