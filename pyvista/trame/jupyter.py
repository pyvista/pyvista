"""Trame utilities for running in Jupyter."""
import asyncio
import logging
import os
import warnings

from IPython import display
from ipywidgets import widgets
from trame.app import get_server
from trame.ui.vuetify import VAppLayout
from trame.widgets import html as html_widgets, vtk as vtk_widgets, vuetify as vuetify_widgets

import pyvista
from pyvista.trame.ui import UI_TITLE, get_or_create_viewer
from pyvista.trame.views import CLOSED_PLOTTER_ERROR

SERVER_DOWN_MESSAGE = """Trame server has not launched.

You must start the trame server before attempting to `show()`
with PyVista.

You can use the following snippet to launch the server:

    from pyvista.trame.jupyter import launch_server
    await launch_server('{name}')

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


class Widget(widgets.HTML):
    """Custom HTML iframe widget for trame viewer."""

    def __init__(self, viewer, src, **kwargs):
        """Initialize."""
        width = kwargs.pop('width', '99%')
        height = kwargs.pop('height', '600px')
        value = f"<iframe src='{src}' style='width: {width}; height: {height};'></iframe>"
        super().__init__(value, **kwargs)
        self._viewer = viewer

    @property
    def viewer(self):
        """Get the associated viewer instance."""
        return self._viewer


def launch_server(server=None):
    """Launch a trame server for use with Jupyter."""
    if server is None:
        server = pyvista.global_theme.trame.jupyter_server_name
    if isinstance(server, str):
        server = get_server(server)

    # Disable serializer errors/warnings for a cleaner output in Jupyter
    # Do this on server launch and not at top level so that it only happens
    # in Jupyter
    if os.environ.get('VTK_DISABLE_SERIALIZER_LOGGER', 'true').lower() == 'true':
        vtk_logger = logging.getLogger("vtkmodules.web.render_window_serializer")
        vtk_logger.disabled = True

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
            # Default to `127.0.0.1` unless user sets TRAME_DEFAULT_HOST
            host=os.environ.get('TRAME_DEFAULT_HOST', '127.0.0.1'),
            port=0,
            open_browser=False,
            show_connection_info=False,
            disable_logging=True,
            timeout=0,
        )
    # else, server is already running
    return server.ready


def build_url(_server, ui=None, server_proxy_enabled=None, server_proxy_prefix=None, **kwargs):
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
        src = f'{kwargs.get("protocol", "http")}://{kwargs.get("host", "localhost")}:{_server.port}/index.html{params}'
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
    default_server_rendering=True,
    return_viewer=False,
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

    return_viewer : bool, optional
        Return the ipywidget.

    **kwargs : dict, optional
        Mostly ignored, though ``protocol`` and ``host`` can be use to
        override the iframe src url and ``hieght`` and ``width`` can be
        used to overrid the iframe style.

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
    )

    # Show as cell result
    src = build_url(
        server,
        ui=plotter._id_name,
        server_proxy_enabled=server_proxy_enabled,
        server_proxy_prefix=server_proxy_prefix,
        **kwargs,
    )

    disp = Widget(viewer, src, **kwargs)
    if return_viewer:
        return disp
    display.display_html(disp)


def elegantly_launch(server):
    """Elegantly launch the Trame server without await.

    This provides a mechanism to launch the Trame Jupyter backend in
    a way that does not require users to await the call.

    .. warning::
        This uses `nest_asyncio <https://github.com/erdewit/nest_asyncio>`_
        which patches the standard lib `asyncio` package and may have
        unintended consequences for some uses cases. We advise strongly to
        make sure PyVista's Jupyter backend is not set to use Trame when not
        in a Jupyter environment.

    """
    try:
        import nest_asyncio
    except ImportError:
        raise ImportError(
            """Please install `nest_asyncio` to automagically launch the trame server without await. Or, to avoid `nest_asynctio` run:

    from pyvista.trame.jupyter import launch_server
    await launch_server()
"""
        )

    async def launch_it():
        await launch_server(server)

    # Basically monkey patches asyncio to support this
    nest_asyncio.apply()

    return asyncio.run(launch_it())
