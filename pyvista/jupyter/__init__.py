"""Jupyter notebook plotting module."""

from __future__ import annotations

import contextlib
import importlib.util
from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args

from typing_extensions import TypeIs

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning as PyVistaDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Callable

JupyterBackendOptions = Literal['static', 'client', 'server', 'trame', 'html', 'none']
ALLOWED_BACKENDS = get_args(JupyterBackendOptions)

_custom_backends: dict[str, Callable[..., object]] = {}
_entry_points_loaded: bool = False


def register_jupyter_backend(name: str, handler: Callable[..., object]) -> None:
    """Register a custom Jupyter backend handler.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    name : str
        Name of the backend (e.g. ``'custom'``). Must not collide with
        a built-in backend name.
    handler : callable
        A callable with signature ``handler(plotter, **kwargs)`` that
        returns an IPython-displayable object.

    Raises
    ------
    ValueError
        If ``name`` collides with a built-in backend.

    Examples
    --------
    >>> import pyvista as pv
    >>> def my_handler(plotter, **kwargs): ...
    >>> pv.register_jupyter_backend('my_backend', my_handler)  # doctest: +SKIP

    """
    name = name.lower()
    if name in ALLOWED_BACKENDS:
        msg = f'Cannot register custom backend "{name}": collides with built-in backend.'
        raise ValueError(msg)
    _custom_backends[name] = handler


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
    handler = _custom_backends.get(name)
    if handler is not None:
        return handler
    _discover_entry_points()
    return _custom_backends.get(name)


def _discover_entry_points() -> None:
    """Scan ``pyvista.jupyter.backends`` entry point group and load handlers."""
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    from importlib.metadata import entry_points  # noqa: PLC0415

    eps = entry_points(group='pyvista.jupyter.backends')
    for ep in eps:
        name = ep.name.lower()
        if name not in ALLOWED_BACKENDS and name not in _custom_backends:
            with contextlib.suppress(Exception):
                _custom_backends[name] = ep.load()


def _is_jupyter_backend(backend: str) -> TypeIs[JupyterBackendOptions]:
    """Return True if backend is allowed jupyter backend."""
    return backend in ALLOWED_BACKENDS


def _validate_jupyter_backend(
    backend: str | None,
) -> str:
    """Validate that a jupyter backend is valid.

    Returns the normalized name of the backend. Raises if the backend is invalid.

    """
    # Must be a string
    if backend is None:
        backend = 'none'
    backend = backend.lower()

    if not importlib.util.find_spec('IPython'):
        msg = 'Install IPython to display with pyvista in a notebook.'
        raise ImportError(msg)

    if _is_jupyter_backend(backend):
        if backend in ['server', 'client', 'trame', 'html']:
            try:
                from pyvista.trame.jupyter import show_trame as show_trame  # noqa: PLC0415
            except ImportError:  # pragma: no cover
                msg = 'Please install trame dependencies: pip install "pyvista[jupyter]"'
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


def set_jupyter_backend(backend, name=None, **kwargs):  # noqa: ARG001
    """Set the plotting backend for a jupyter notebook.

    Parameters
    ----------
    backend : str
        Jupyter backend to use when plotting.  Must be one of the following:

        * ``'static'`` : Display a single static image within the
          Jupyterlab environment.  Still requires that a virtual
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
          instead display using dedicated VTK render windows.  This
          will generate nothing on headless servers even with a
          virtual framebuffer.

        Custom backends registered via
        :func:`register_jupyter_backend` are also accepted.

    name : str, optional
        The unique name identifier for the server.
    **kwargs : dict, optional
        Any additional keyword arguments to pass to the server launch.

    Examples
    --------
    Enable the trame Trame backend.

    >>> pv.set_jupyter_backend('trame')  # doctest:+SKIP

    Just show static images.

    >>> pv.set_jupyter_backend('static')  # doctest:+SKIP

    Disable all plotting within JupyterLab and display using a
    standard desktop VTK render window.

    >>> pv.set_jupyter_backend(None)  # doctest:+SKIP

    """
    pv.global_theme._jupyter_backend = _validate_jupyter_backend(backend)
