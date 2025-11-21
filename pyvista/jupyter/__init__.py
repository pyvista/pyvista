"""Jupyter notebook plotting module."""

from __future__ import annotations

import importlib.util
from typing import Literal
from typing import get_args

from typing_extensions import TypeIs

import pyvista as pv
from pyvista.core.errors import PyVistaDeprecationWarning as PyVistaDeprecationWarning

JupyterBackendOptions = Literal['static', 'client', 'server', 'trame', 'html', 'none']
"""Jupyter backend to use."""

ALLOWED_BACKENDS = get_args(JupyterBackendOptions)


def _is_jupyter_backend(backend: str) -> TypeIs[JupyterBackendOptions]:
    """Return True if backend is allowed jupyter backend."""
    return backend in ALLOWED_BACKENDS


def _validate_jupyter_backend(
    backend: str | None,
) -> JupyterBackendOptions:
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

    if not _is_jupyter_backend(backend):
        backend_list_str = ', '.join([f'"{item}"' for item in ALLOWED_BACKENDS])
        msg = (
            f'Invalid Jupyter notebook plotting backend "{backend}".\n'
            f'Use one of the following:\n{backend_list_str}'
        )
        raise ValueError(msg)

    if backend in ['server', 'client', 'trame', 'html']:
        try:
            from pyvista.trame.jupyter import show_trame as show_trame  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            msg = 'Please install trame dependencies: pip install "pyvista[jupyter]"'
            raise ImportError(msg)

    return backend


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
