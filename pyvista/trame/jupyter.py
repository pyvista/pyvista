import asyncio
import logging

from trame.app import get_server, jupyter
from trame_vtk.modules import vtk as vtk_module

from pyvista.trame import CLOSED_PLOTTER_ERROR, ui

logger = logging.getLogger(__name__)


def _show(_server, ui=None, server_proxy=False, server_proxy_prefix=None, **kwargs):
    """
    Helper function to show a server ui element into the cell.

    :param _server: the server on which the UI is defined
    :type _server: trame_server.core.Server

    :param ui: the name of the ui section to display. (Default: 'main')
    :type ui: str

    :param server_proxy: build the URL relative for use with `jupyter-server-proxy`
    :type server_proxy: bool

    :param server_proxy_prefix: URL prefix when using `jupyter-server-proxy` on JupyterHub instances.
    :type server_proxy: str

    :param **kwargs: any keyword arguments are pass to the Jupyter IFrame.
        Additionally `protocol=` and `host=` can be use to override the iframe src url.
    """
    if isinstance(_server, str):
        _server = get_server(_server)

    def on_ready(**_):
        params = f"?ui={ui}" if ui else ""
        if server_proxy:
            src = f"{server_proxy_prefix + '/' if server_proxy_prefix else ''}proxy/{_server.port}/index.html{params}"
        else:
            src = f"{kwargs.get('protocol', 'http')}://{kwargs.get('host', 'localhost')}:{_server.port}/index.html{params}"
        loop = asyncio.get_event_loop()
        loop.call_later(0.1, lambda: jupyter.display_iframe(src, **kwargs))
        _server.controller.on_server_ready.discard(on_ready)

    if _server._running_stage == 0:
        _server.controller.on_server_ready.add(on_ready)
        _server.start(
            exec_mode="task",
            port=0,
            open_browser=False,
            show_connection_info=False,
            disable_logging=True,
            timeout=0,
        )
    elif _server._running_stage == 1:
        _server.controller.on_server_ready.add(on_ready)
    elif _server._running_stage == 2:
        on_ready()


def show_trame(
    plotter,
    server=None,
    local_rendering=True,
    server_proxy=False,
    server_proxy_prefix=None,
    **kwargs,
):
    """Run and display the trame application in jupyter's event loop

    The kwargs are forwarded to IPython.display.IFrame()
    """
    if plotter.render_window is None:
        raise RuntimeError(CLOSED_PLOTTER_ERROR)

    if server is None:
        server = get_server()

    if isinstance(server, str):
        server = get_server(server)

        # Needed to support multi-instance in Jupyter
        server.enable_module(vtk_module)

    # Disable logging
    logger.setLevel(logging.WARNING)

    # Initialize app
    ui.initialize(server, plotter, local_rendering=local_rendering)

    # Show as cell result
    _show(server, server_proxy=server_proxy, server_proxy_prefix=server_proxy_prefix, **kwargs)
