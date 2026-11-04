"""Jupyter notebook plotting module."""

from __future__ import annotations

# ``Callable`` is used in the ``JupyterBackendRegistration``
# NamedTuple field annotations and must be available at runtime so that
# ``typing.get_type_hints`` (called by Sphinx autodoc) can resolve them.
from collections.abc import Callable  # noqa: TC003
import importlib.util
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import get_args

from typing_extensions import TypeIs

import pyvista as pv
from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning as PyVistaDeprecationWarning
from pyvista.core.utilities._registry import _HandlerRegistry

_TrameServerBackendOptions = Literal['client', 'server', 'trame']  # served in an iframe
_TrameBackendOptions = Literal[_TrameServerBackendOptions, 'html']
JupyterBackendOptions = Literal['static', _TrameBackendOptions, 'none']
ALLOWED_BACKENDS = get_args(JupyterBackendOptions)
_TRAME_BACKENDS = get_args(_TrameBackendOptions)
_TRAME_SERVER_BACKENDS = get_args(_TrameServerBackendOptions)

JUPYTER_BACKEND_ENTRY_POINT_GROUP = 'pyvista.jupyter_backends'


_registry: _HandlerRegistry[Callable[..., object]] = _HandlerRegistry(
    JUPYTER_BACKEND_ENTRY_POINT_GROUP,
    label='Jupyter backend',
    builtin=lambda name: name in ALLOWED_BACKENDS,
    builtin_label='built-in backend',
)
_custom_backends = _registry.explicit
_plugins = _registry.plugins
_save_registry_state = _registry.snapshot
_restore_registry_state = _registry.restore


class JupyterBackendRegistration(NamedTuple):
    """Describe one registered custom Jupyter backend.

    Returned by :func:`~pyvista.registered_jupyter_backends`.

    .. versionadded:: 0.48.0

    Attributes
    ----------
    name : str
        Backend name.
    handler : callable
        The backend handler callable.
    source : str
        Human-readable origin in the form ``'module.qualname'`` for
        explicit registrations or the entry-point ``value`` for
        plugin-discovered registrations.

    """

    name: str
    handler: Callable[..., object]
    source: str


def register_jupyter_backend(
    name: str,
    handler: Callable[..., object],
    *,
    override: bool = False,
) -> None:
    """Register a custom Jupyter backend handler.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    name : str
        Name of the backend (for example, ``'custom'``). Must not collide with
        a built-in backend name unless ``override=True`` is passed.
    handler : callable
        A callable with signature ``handler(plotter, **kwargs)`` that
        returns an IPython-displayable object.
    override : bool, default: False
        If ``True``, allow registering a name that collides with a
        built-in backend. Also silences the warning emitted when
        replacing an existing custom registration.

    Raises
    ------
    ValueError
        If ``name`` collides with a built-in backend and ``override``
        is ``False``.

    Warns
    -----
    UserWarning
        If ``name`` already refers to a registered custom backend. The
        new registration replaces the old one (last wins); pass
        ``override=True`` to silence the warning.

    Examples
    --------
    >>> import pyvista as pv
    >>> def my_handler(plotter, **kwargs): ...
    >>> pv.register_jupyter_backend('my_backend', my_handler)  # doctest: +SKIP

    """
    _registry.register(name, handler, override=override)


def registered_jupyter_backends() -> tuple[JupyterBackendRegistration, ...]:
    """Return every custom Jupyter backend currently registered.

    Forces discovery of any pending entry-point plugins so the returned
    list reflects every backend visible to PyVista. A plugin that fails
    to import emits a ``UserWarning`` and is skipped; the rest still
    appear in the result.

    .. versionadded:: 0.48.0

    .. versionchanged:: 0.50.0
        A plugin whose import failed stays pending and is retried on
        every call, so a backend whose dependency was installed after
        the failure becomes available again.

    Returns
    -------
    tuple[JupyterBackendRegistration, ...]
        One record per registered backend. Each record exposes
        ``name``, ``handler``, and ``source``. Built-in backends are
        not included; only custom registrations.

    """
    return tuple(JupyterBackendRegistration(*record) for record in _registry.registered())


def _get_custom_backend_handler(name: str) -> Callable[..., object] | None:
    """Look up a custom backend handler by name.

    Checks the registry first, then discovers entry points lazily.

    Parameters
    ----------
    name : str
        Backend name to look up.

    Returns
    -------
    callable or None
        The handler if found, otherwise ``None``.

    """
    _plugins.ensure_loaded()
    return _registry.get(name)


def _resolve_backend() -> str:
    """Auto-detect the best available Jupyter backend.

    Priority: registered custom backends > trame > static.

    Returns
    -------
    str
        Name of the best available backend.

    """
    _plugins.ensure_loaded()
    # Prefer user registrations over plugin-discovered backends; among
    # plugin-discovered ones, prefer 'trame' (the all-in-one backend)
    # over the more specialized 'server'/'client'/'html' aliases.
    for name in _custom_backends:
        if ':' not in _custom_backends.source(name):  # explicit registration (module.qualname)
            return name
    if 'trame' in _custom_backends:
        return 'trame'
    if _custom_backends:
        return next(iter(_custom_backends))
    return 'static'


def _is_jupyter_backend(backend: str) -> TypeIs[JupyterBackendOptions]:
    """Return True if backend is allowed jupyter backend."""
    return backend in ALLOWED_BACKENDS


def _validate_jupyter_backend(
    backend: str | None,
) -> str | None:
    """Validate that a jupyter backend is valid.

    Returns the normalized name of the backend, or ``None`` to indicate that
    the backend should be auto-detected at display time. Raises if the backend
    is invalid.

    """
    # ``None`` is the auto-detect sentinel; preserve it
    if backend is None:
        return None
    backend = backend.lower()

    if not importlib.util.find_spec('IPython'):
        msg = 'Install IPython to display with pyvista in a notebook.'
        raise ImportError(msg)

    if _is_jupyter_backend(backend):
        if backend in _TRAME_BACKENDS:
            if _get_custom_backend_handler(backend) is None:  # pragma: no cover
                msg = 'Please install trame dependencies: pip install pyvista[jupyter]'
                raise ImportError(msg)
        return backend

    # Check custom backends (registry + entry points)
    if _get_custom_backend_handler(backend) is not None:
        return backend

    all_backends = list(ALLOWED_BACKENDS) + sorted(_custom_backends.keys())
    backend_list_str = ', '.join([f'"{item}"' for item in all_backends])
    msg = (
        f'Invalid Jupyter notebook plotting backend "{backend}".\n'
        f'Use one of the following:\n{backend_list_str}'
    )
    raise ValueError(msg)


def set_jupyter_backend(
    backend: JupyterBackendOptions | str | None, name: str | None = None, **kwargs: Any
) -> None:
    """Set the plotting backend for a jupyter notebook.

    Parameters
    ----------
    backend : str
        Jupyter backend to use when plotting. Must be one of the following:

        * ``'static'`` : Display a single static image within the
          Jupyterlab environment. Still requires that a virtual
          framebuffer be set up when displaying on a headless server,
          but does not require any additional modules to be installed.

        * ``'client'`` : Export/serialize the scene graph to be rendered
          with VTK.js client-side through ``trame``. Requires ``trame``
          and ``jupyter-server-proxy`` to be installed.

        * ``'server'``: Render remotely and stream the resulting VTK
          images back to the client using ``trame``. This replaces the
          ``'ipyvtklink'`` backend with better performance.
          Supports the most VTK features, but suffers from minor lag due
          to remote rendering. Requires that a virtual framebuffer be set
          up when displaying on a headless server. Must have at least ``trame``
          and ``jupyter-server-proxy`` installed for cloud/remote Jupyter
          instances. This mode is also aliased by ``'trame'``.

        * ``'trame'``: The full Trame-based backend that combines both
          ``'server'`` and ``'client'`` into one backend. This requires a
          virtual frame buffer.

        * ``'html'`` : Export/serialize the scene graph to be rendered
          with the Trame client backend but in a static HTML file.

        * ``'none'`` : Do not display any plots within jupyterlab,
          instead display using dedicated VTK render windows. This
          will generate nothing on headless servers even with a
          virtual framebuffer.

        Custom backends registered via :func:`~pyvista.register_jupyter_backend`
        are also accepted. Pass ``None`` to reset to auto-detection at display
        time.

    name : str, optional
        Unused.

        .. deprecated:: 0.49
            This parameter has never had any effect and will be removed.

    **kwargs : dict, optional
        Unused.

        .. deprecated:: 0.49
            These parameters have never had any effect and will be removed.

    Examples
    --------
    Enable the trame Trame backend.

    >>> pv.set_jupyter_backend('trame')  # doctest:+SKIP

    Just show static images.

    >>> pv.set_jupyter_backend('static')  # doctest:+SKIP

    Disable all plotting within JupyterLab and display using a
    standard desktop VTK render window.

    >>> pv.set_jupyter_backend('none')  # doctest:+SKIP

    Reset to auto-detect the best available backend.

    >>> pv.set_jupyter_backend(None)  # doctest:+SKIP

    """
    if name is not None or kwargs:
        warn_external(
            'The `name` and `**kwargs` arguments of `set_jupyter_backend` are unused and '
            'will be removed. Remove them from the call.',
            PyVistaDeprecationWarning,
        )
    pv.global_theme._jupyter_backend = _validate_jupyter_backend(backend)
