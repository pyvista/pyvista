"""Trame utilities for running in Jupyter."""
import logging
import warnings

from IPython import display
from trame.app import get_server
from trame.widgets import html as html_widgets, vtk as vtk_widgets, vuetify as vuetify_widgets

import pyvista
from pyvista.trame.ui import initialize
from pyvista.trame.views import CLOSED_PLOTTER_ERROR

SERVER_DOWN_MESSAGE = """Trame server has not launched.

You must start the trame server before attempting to `show()`
with PyVista.

You can use the following snippet to launch the server:

    from pyvista.trame.jupyter import launch_server
    await launch_server('{name}')

"""
JUPYTER_SERVER_DOWN_MESSAGE = """Trame server has not launched.

Prior to plotting, please make sure to run `await set_jupyter_backend('trame')` when using the `'trame'`, `'server'`, or `'client'` Jupyter backends.

    import pyvista as pv
    await pv.set_jupyter_backend('trame')

It is critial that this `await` is used.

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


def launch_server(server):
    """Launch a trame server."""
    if isinstance(server, str):
        server = get_server(server)

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
            port=0,
            open_browser=False,
            show_connection_info=False,
            disable_logging=True,
            timeout=0,
        )
    # else, server is already running
    return server.ready


def build_iframe(_server, ui=None, server_proxy_enabled=None, server_proxy_prefix=None, **kwargs):
    """Build IPython display.IFrame object for the trame view."""
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
        src = f'{kwargs.get("protocol", "http")}://{kwargs.get("host", "localhost")}:{_server.port}/index.html{params}'
    iframe_kwargs = {
        'width': '100%',
        'height': 600,
    }
    iframe_kwargs.update(**kwargs)
    logger.debug(src)
    return display.IFrame(src=src, **iframe_kwargs)


def show_trame(
    plotter,
    mode=None,
    name=None,
    server_proxy_enabled=None,
    server_proxy_prefix=None,
    collapse_menu=False,
    default_server_rendering=True,
    **kwargs,
):
    """Run and display the trame application in jupyter's event loop.

    plotter : pyvista.BasePlotter
        The PyVista plotter to show.

    mode : str, optional
        The UI view mode. This can be set on the global theme. Options are:
            * ``'trame'``: Uses a view that can switch between client and server
              rendering modes.
            * ``'server'``: Uses a view that is purely server rendering.
            * ``'client'``: Uses a view that is purely client rendering (generally
              safe without a virtual frame buffer)

    name : str, optional
        The name of the trame server on which the UI is defined

    server_proxy_enabled : bool, default: False
        Build a relative URL for use with ``jupyter-server-proxy``.

    server_proxy_prefix : str, optional
        URL prefix when using ``server_proxy_enabled``. This can be set globally in
        the theme. To ignore, pass ``False``. For use with
        ``jupyter-server-proxy``, often set to ``proxy/``.

    collapse_menu : bool, default: False
        Collapse the UI menu (camera controls, etc.) on start.

    default_server_rendering : bool, default: True
        Whether to use server-side or client-side rendering on-start when
        using the ``'trame'`` mode.

    **kwargs : dict, optional
        any keyword arguments are pass to the Jupyter IFrame. Additionally
        `protocol=` and `host=` can be use to override the iframe src url.
    """
    if plotter.render_window is None:
        raise RuntimeError(CLOSED_PLOTTER_ERROR)

    if name is None:
        server = get_server(name=pyvista.global_theme.trame.jupyter_server_name)
    else:
        server = get_server(name=name)
    if name is None and not server.running:
        raise TrameJupyterServerDownError()
    elif not server.running:
        raise TrameServerDownError(name)

    # Initialize app
    ui_name = initialize(
        server,
        plotter,
        mode=mode,
        default_server_rendering=default_server_rendering,
        collapse_menu=collapse_menu,
    )

    # Show as cell result
    return build_iframe(
        server,
        ui=ui_name,
        server_proxy_enabled=server_proxy_enabled,
        server_proxy_prefix=server_proxy_prefix,
        **kwargs,
    )
