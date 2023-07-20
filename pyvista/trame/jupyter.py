"""Trame utilities for running in Jupyter."""
import asyncio
import logging
import os
import warnings

from trame.app import get_server
from trame.ui.vuetify import VAppLayout
from trame.widgets import html as html_widgets, vtk as vtk_widgets, vuetify as vuetify_widgets

try:
    from ipywidgets.widgets import HTML
except ImportError:
    HTML = object


import pyvista
from pyvista.trame.ui import UI_TITLE, get_or_create_viewer
from pyvista.trame.views import CLOSED_PLOTTER_ERROR

SERVER_DOWN_MESSAGE = """Trame server has not launched.

You must start the trame server before attempting to `show()`
with PyVista.

You can use the following snippet to launch the server:

    from pyvista.trame.jupyter import launch_server
    await launch_server('{name}').ready

"""
JUPYTER_SERVER_DOWN_MESSAGE = """Trame server has not launched.

Prior to plotting, please make sure to run `set_jupyter_backend('trame')` when using the `'trame'`, `'server'`, or `'client'` Jupyter backends.

    import pyvista as pv
    pv.set_jupyter_backend('trame')

If this issue persists, please open an issue in PyVista: https://github.com/pyvista/pyvista/issues

"""

logger = logging.getLogger(__name__)


class TrameServerDownError(RuntimeError):
    """Exception when trame server is down for Jupyter."""

    def __init__(self, server_name):
        """Call the base class constructor with the custom message."""
        super().__init__(SERVER_DOWN_MESSAGE.format(name=server_name))


class TrameJupyterServerDownError(RuntimeError):
    """Exception when trame server is down for Jupyter."""

    def __init__(self):
        """Call the base class constructor with the custom message."""
        # Be incredibly verbose on how users should launch trame server
        # Both warn so it appears at top
        warnings.warn(JUPYTER_SERVER_DOWN_MESSAGE)
        # and Error
        super().__init__(JUPYTER_SERVER_DOWN_MESSAGE)


class Widget(HTML):
    """Custom HTML iframe widget for trame viewer."""

    def __init__(self, viewer, src, width, height, **kwargs):
        """Initialize."""
        if HTML is object:
            raise ImportError('Please install `ipywidgets`.')
        value = f"<iframe src='{src}' style='width: {width}; height: {height};'></iframe>"
        super().__init__(value, **kwargs)
        self._viewer = viewer
        self._src = src

    @property
    def viewer(self):
        """Get the associated viewer instance."""
        return self._viewer

    @property
    def src(self):
        """Get the src URL."""
        return self._src


def launch_server(server=None, port=None, host=None):
    """Launch a trame server for use with Jupyter.

    Parameters
    ----------
    server : str, optional
        By default this uses :attr:`pyvista.global_theme.trame.jupyter_server_name
        <pyvista.plotting.themes._TrameConfig.jupyter_server_name>`, which by default is
        set to ``'pyvista-jupyter'``.

        If a server name is given and such server is not available yet, it will
        be created otherwise the previously created instance will be returned.

    port : int, optional
        The port on which to bind the server. Defaults to 0 to automatically
        find an available port.

    host : str, optional
        The host name to bind the server to on launch. Server will bind to
        ``127.0.0.1`` by default unless user sets the environment variable ``TRAME_DEFAULT_HOST``.

    Returns
    -------
    trame_server.core.Server
        The launched Trame server. To ``await`` the launch, use the
        ``.ready`` future attribute on the server.

    """
    if server is None:
        server = pyvista.global_theme.trame.jupyter_server_name
    if isinstance(server, str):
        server = get_server(server)
    if port is None:
        port = pyvista.global_theme.trame.jupyter_server_port
    if host is None:
        # Default to `127.0.0.1` unless user sets TRAME_DEFAULT_HOST
        host = os.environ.get('TRAME_DEFAULT_HOST', '127.0.0.1')

    # Must enable all used modules
    html_widgets.initialize(server)
    vtk_widgets.initialize(server)
    vuetify_widgets.initialize(server)

    def on_ready(**_):
        logger.debug(f'Server ready: {server}')

    if server._running_stage == 0:
        server.controller.on_server_ready.add(on_ready)
        server.start(
            exec_mode='task',
            host=host,
            port=port,
            open_browser=False,
            show_connection_info=False,
            disable_logging=True,
            timeout=0,
        )
    # else, server is already running or launching
    return server


def build_url(
    _server,
    ui=None,
    server_proxy_enabled=None,
    server_proxy_prefix=None,
    host='localhost',
    protocol='http',
):
    """Build the URL for the iframe."""
    params = f'?ui={ui}&reconnect=auto' if ui else '?reconnect=auto'
    if server_proxy_enabled is None:
        server_proxy_enabled = pyvista.global_theme.trame.server_proxy_enabled
    if server_proxy_enabled:
        if server_proxy_prefix is None:
            server_proxy_prefix = pyvista.global_theme.trame.server_proxy_prefix
        # server_proxy_prefix assumes trailing slash
        src = (
            f"{server_proxy_prefix if server_proxy_prefix else ''}{_server.port}/index.html{params}"
        )
    else:
        src = f'{protocol}://{host}:{_server.port}/index.html{params}'
    logger.debug(src)
    return src


def initialize(
    server, plotter, mode=None, default_server_rendering=True, collapse_menu=False, **kwargs
):
    """Generate the UI for a given plotter."""
    state = server.state
    state.trame__title = UI_TITLE

    viewer = get_or_create_viewer(plotter, suppress_rendering=mode == 'client')

    with VAppLayout(server, template_name=plotter._id_name):
        viewer.ui(
            mode=mode,
            default_server_rendering=default_server_rendering,
            collapse_menu=collapse_menu,
            **kwargs,
        )

    return viewer


def show_trame(
    plotter,
    mode=None,
    name=None,
    server_proxy_enabled=None,
    server_proxy_prefix=None,
    collapse_menu=False,
    add_menu=True,
    default_server_rendering=True,
    handler=None,
    **kwargs,
):
    """Run and display the trame application in jupyter's event loop.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The PyVista plotter to show.

    mode : str, optional
        The UI view mode. This can be set on the global theme. Options are:

        * ``'trame'``: Uses a view that can switch between client and server
          rendering modes.
        * ``'server'``: Uses a view that is purely server rendering.
        * ``'client'``: Uses a view that is purely client rendering (generally
          safe without a virtual frame buffer)

    name : str, optional
        The name of the trame server on which the UI is defined.

    server_proxy_enabled : bool, default: False
        Build a relative URL for use with ``jupyter-server-proxy``.

    server_proxy_prefix : str, optional
        URL prefix when using ``server_proxy_enabled``. This can be set
        globally in the theme. To ignore, pass ``False``. For use with
        ``jupyter-server-proxy``, often set to ``proxy/``.

    collapse_menu : bool, default: False
        Collapse the UI menu (camera controls, etc.) on start.

    add_menu : bool, default: True
        Add a UI controls VCard to the VContainer.

    default_server_rendering : bool, default: True
        Whether to use server-side or client-side rendering on-start when
        using the ``'trame'`` mode.

    handler : callable, optional
        Pass a callable that accptes the viewer instance, the string URL,
        and ``**kwargs`` to create custom HTML representations of the output.

        .. code:: python

            import pyvista as pv
            from IPython.display import IFrame

            mesh = pv.Wavelet()


            def handler(viewer, src, **kwargs):
                return IFrame(src, '75%', '500px')


            p = pv.Plotter(notebook=True)
            _ = p.add_mesh(mesh)
            iframe = p.show(
                jupyter_backend='trame',
                jupyter_kwargs=dict(handler=handler),
                return_viewer=True,
            )
            iframe

    **kwargs : dict, optional
        Mostly ignored, though ``protocol`` and ``host`` can be use to
        override the iframe src url and ``height`` and ``width`` can be
        used to override the iframe style. Remaining kwargs are passed to
        ``ipywidgets.widgets.HTML``.

    Returns
    -------
    ipywidgets.widgets.HTML or handler result
        Returns a HTML IFrame widget or the result of the passed handler.

    """
    if plotter.render_window is None:
        raise RuntimeError(CLOSED_PLOTTER_ERROR)

    if name is None:
        server = get_server(name=pyvista.global_theme.trame.jupyter_server_name)
    else:
        server = get_server(name=name)
    if name is None and not server.running:
        elegantly_launch(server)
        if not server.running:
            raise TrameJupyterServerDownError()
    elif not server.running:
        raise TrameServerDownError(name)

    # Initialize app
    viewer = initialize(
        server,
        plotter,
        mode=mode,
        default_server_rendering=default_server_rendering,
        collapse_menu=collapse_menu,
        add_menu=add_menu,
    )

    # Show as cell result
    src = build_url(
        server,
        ui=plotter._id_name,
        server_proxy_enabled=server_proxy_enabled,
        server_proxy_prefix=server_proxy_prefix,
        host=kwargs.get('host', 'localhost'),
        protocol=kwargs.get('protocol', 'http'),
    )

    if plotter._window_size_unset:
        dw, dh = '99%', '600px'
    else:
        dw, dh = plotter.window_size
        dw = f'{dw}px'
        dh = f'{dh}px'
    kwargs.setdefault('width', dw)
    kwargs.setdefault('height', dh)

    if callable(handler):
        return handler(viewer, src, **kwargs)
    return Widget(viewer, src, **kwargs)


def elegantly_launch(*args, **kwargs):
    """Elegantly launch the Trame server without await.

    This provides a mechanism to launch the Trame Jupyter backend in
    a way that does not require users to await the call.

    This is a thin wrapper of
    :func:`launch_server() <pyvista.trame.jupyter.launch_server>`.

    Returns
    -------
    trame_server.core.Server
        The launched trame server.

    Warnings
    --------
    This uses `nest_asyncio <https://github.com/erdewit/nest_asyncio>`_ which
    patches the standard lib `asyncio` package and may have unintended
    consequences for some uses cases. We advise strongly to make sure PyVista's
    Jupyter backend is not set to use Trame when not in a Jupyter environment.

    """
    try:
        import nest_asyncio
    except ImportError:
        raise ImportError(
            """Please install `nest_asyncio` to automagically launch the trame server without await. Or, to avoid `nest_asynctio` run:

    from pyvista.trame.jupyter import launch_server
    await launch_server().ready
"""
        )

    async def launch_it():
        await launch_server(*args, **kwargs).ready

    # Basically monkey patches asyncio to support this
    nest_asyncio.apply()

    return asyncio.run(launch_it())
