"""Trame utilities for running in Jupyter."""
import logging
import warnings

from IPython import display
from trame.app import get_server
from trame.widgets import vtk as vtk_widgets, vuetify as vuetify_widgets

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

Prior to plotting, please make sure to run `await set_jupyter_backend('server')` when using the `'server'` or `'client'` Jupyter backends.

    import pyvista as pv
    await pv.set_jupyter_backend('server')

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


def is_server_up(server):
    """Check if server is running."""
    if isinstance(server, str):
        server = get_server(server)
    return server._running_stage >= 2


def launch_server(server):
    """Launch a trame server."""
    if isinstance(server, str):
        server = get_server(server)

    # Must enable all used modules
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


def build_iframe(_server, ui=None, relative_url=None, relative_url_prefix=None, **kwargs):
    """Build IPython display.IFrame object for the trame view."""
    params = f'?ui={ui}&reconnect=auto' if ui else '?reconnect=auto'
    if relative_url is None:
        relative_url = pyvista.global_theme.trame.relative_url_enabled
    if relative_url:
        if relative_url_prefix is None:
            relative_url_prefix = pyvista.global_theme.trame.relative_url_prefix
        # relative_url_prefix assumes trailing slash
        src = (
            f"{relative_url_prefix if relative_url_prefix else ''}{_server.port}/index.html{params}"
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
    local_rendering=False,
    name=None,
    relative_url=None,
    relative_url_prefix=None,
    **kwargs,
):
    """Run and display the trame application in jupyter's event loop.

    plotter : pyvista.BasePlotter
        The PyVista plotter to show.

    local_rendering : bool, default: False
        Whether to use local (client-side) rendering.

    name : str
        The name of the trame server on which the UI is defined

    relative_url : bool, default: False
        Build a relative URL. Often for use with ``jupyter-server-proxy``.

    relative_url_prefix : str, optional
        URL prefix when using ``relative_url``. This can be set globally in
        the theme. To ignore, pass ``False``. For use with
        ``jupyter-server-proxy``, often set to ``proxy/``.

    **kwargs
        any keyword arguments are pass to the Jupyter IFrame. Additionally
        `protocol=` and `host=` can be use to override the iframe src url.
    """
    if plotter.render_window is None:
        raise RuntimeError(CLOSED_PLOTTER_ERROR)

    if name is None:
        server = get_server(name=pyvista.global_theme.trame.jupyter_server_name)
    else:
        server = get_server(name=name)
    if name is None and not is_server_up(server):
        raise TrameJupyterServerDownError()
    elif not is_server_up(server):
        raise TrameServerDownError(name)

    # Initialize app
    ui_name = initialize(server, plotter, local_rendering=local_rendering)

    # Show as cell result
    return build_iframe(
        server,
        ui=ui_name,
        relative_url=relative_url,
        relative_url_prefix=relative_url_prefix,
        **kwargs,
    )
