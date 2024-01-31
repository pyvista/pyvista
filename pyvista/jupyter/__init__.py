"""Jupyter notebook plotting module."""
# flake8: noqa: F401

import warnings

import pyvista
from pyvista.core.errors import PyVistaDeprecationWarning

ALLOWED_BACKENDS = [
    'static',
    'client',
    'server',
    'trame',
    'html',
    'none',
]


def _validate_jupyter_backend(backend):
    """Validate that a jupyter backend is valid.

    Returns the normalized name of the backend. Raises if the backend is invalid.

    """
    # Must be a string
    if backend is None:
        backend = 'none'
    backend = backend.lower()

    try:
        import IPython
    except ImportError:  # pragma: no cover
        raise ImportError('Install IPython to display with pyvista in a notebook.')

    if backend not in ALLOWED_BACKENDS:
        backend_list_str = ', '.join([f'"{item}"' for item in ALLOWED_BACKENDS])
        raise ValueError(
            f'Invalid Jupyter notebook plotting backend "{backend}".\n'
            f'Use one of the following:\n{backend_list_str}'
        )

    if backend in ['server', 'client', 'trame', 'html']:
        try:
            from pyvista.trame.jupyter import show_trame
        except ImportError:  # pragma: no cover
            raise ImportError('Please install `trame` and `ipywidgets` to use this feature.')

    if backend == 'none':
        backend = None
    return backend


def set_jupyter_backend(backend, name=None, **kwargs):
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
    pyvista.global_theme._jupyter_backend = _validate_jupyter_backend(backend)
