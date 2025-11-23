"""Trame utilities for running in Jupyter."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING
from typing import Concatenate
from typing import Literal
import warnings

from trame.widgets import html as html_widgets
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify as vuetify2_widgets
from trame.widgets import vuetify3 as vuetify3_widgets

try:
    from ipywidgets.widgets import HTML
except ImportError:
    HTML = object


import pyvista as pv
from pyvista.trame.ui import UI_TITLE
from pyvista.trame.ui import get_viewer
from pyvista.trame.views import CLOSED_PLOTTER_ERROR
from pyvista.trame.views import get_server

if TYPE_CHECKING:
    from collections.abc import Callable

    from IPython.display import IFrame

    from pyvista.jupyter import JupyterBackendOptions
    from pyvista.plotting.plotter import Plotter
    from pyvista.trame.ui.vuetify2 import Viewer

SERVER_DOWN_MESSAGE = """Trame server has not launched.

You must start the trame server before attempting to `show()`
with PyVista.

You can use the following snippet to launch the server:

    from pyvista.trame.jupyter import launch_server
    await launch_server('{name}').ready

"""
JUPYTER_SERVER_DOWN_MESSAGE = """Trame server has not launched.

Prior to plotting, please make sure to run `set_jupyter_backend('trame')` when using the
`'trame'`, `'server'`, or `'client'` Jupyter backends.

    import pyvista as pv
    pyvista.set_jupyter_backend('trame')

If this issue persists, please open an issue in PyVista: https://github.com/pyvista/pyvista/issues

"""

logger = logging.getLogger(__name__)


class TrameServerDownError(RuntimeError):  # numpydoc ignore=PR01
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
        warnings.warn(JUPYTER_SERVER_DOWN_MESSAGE, stacklevel=2)  # pragma: no cover
        # and Error
        super().__init__(JUPYTER_SERVER_DOWN_MESSAGE)


class Widget(HTML):  # type: ignore[misc]  # numpydoc ignore=PR01
    """Custom HTML iframe widget for trame viewer."""

    def __init__(self, viewer, src, width=None, height=None, iframe_attrs=None, **kwargs):
        """Initialize."""
        if HTML is object:
            msg = 'Please install `ipywidgets`.'
            raise ImportError(msg)
        # eventually we could maybe expose this, but for now make sure we're at least
        # consistent with matplotlib's color (light gray)

        if iframe_attrs is None:
            iframe_attrs = {}

        border = 'border: 1px solid rgb(221,221,221);'

        iframe_attrs = {
            **iframe_attrs,
            'src': src,
            'class': 'pyvista',
            'style': f'width: {width}; height: {height}; {border}',
        }

        iframe_attrs_str = ' '.join(f'{key}="{value!s}"' for key, value in iframe_attrs.items())

        value = f'<iframe {iframe_attrs_str}></iframe>'

        super().__init__(value, **kwargs)
        self._viewer = viewer
        self._src = src

    @property
    def viewer(self):  # numpydoc ignore=RT01
        """Get the associated viewer instance."""
        return self._viewer

    @property
    def src(self):  # numpydoc ignore=RT01
        """Get the src URL."""
        return self._src


class EmbeddableWidget(HTML):  # type: ignore[misc]  # numpydoc ignore=PR01
    """Custom HTML iframe widget for embedding the trame viewer."""

    def __init__(self, plotter, width, height, **kwargs):
        """Initialize."""
        if HTML is object:
            msg = 'Please install `ipywidgets`.'
            raise ImportError(msg)
        scene = plotter.export_html(filename=None)
        src = scene.getvalue().replace('"', '&quot;')
        # eventually we could maybe expose this, but for now make sure we're at least
        # consistent with matplotlib's color (light gray)
        border = 'border: 1px solid rgb(221,221,221);'
        value = (
            f'<iframe srcdoc="{src}" class="pyvista" style="width: {width}; '
            f'height: {height}; {border}"></iframe>'
        )
        super().__init__(value, **kwargs)
        self._src = src


def launch_server(server=None, port=None, host=None, wslink_backend=None, **kwargs):
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

    wslink_backend : str, optional
        The wslink backend that the server should use
        ``aiohttp`` by default, ``jupyter`` if the
        `trame_jupyter_extension <https://github.com/Kitware/trame-jupyter-extension>`_
        is used.

    **kwargs : dict, optional
        Any additional keyword arguments to pass to ``pyvista.trame.views.get_server``.

    Returns
    -------
    trame_server.core.Server
        The launched Trame server. To ``await`` the launch, use the
        ``.ready`` future attribute on the server.

    """
    if server is None:
        server = pv.global_theme.trame.jupyter_server_name
    if isinstance(server, str):
        server = get_server(server, **kwargs)
    if port is None:
        port = pv.global_theme.trame.jupyter_server_port
    if host is None:
        # Default to `127.0.0.1` unless user sets TRAME_DEFAULT_HOST
        host = os.environ.get('TRAME_DEFAULT_HOST', '127.0.0.1')
    if (
        wslink_backend is None and pv.global_theme.trame.jupyter_extension_enabled
    ):  # pragma: no cover
        wslink_backend = 'jupyter'

    # Must enable all used modules
    html_widgets.initialize(server)
    vtk_widgets.initialize(server)

    if server.client_type == 'vue2':
        vuetify2_widgets.initialize(server)
    else:
        vuetify3_widgets.initialize(server)

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
            backend=wslink_backend,
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
):  # numpydoc ignore=PR01,RT01
    """Build the URL for the iframe."""
    params = f'?ui={ui}&reconnect=auto' if ui else '?reconnect=auto'
    if server_proxy_enabled is None:
        server_proxy_enabled = pv.global_theme.trame.server_proxy_enabled
    if server_proxy_enabled:
        if server_proxy_prefix is None:
            server_proxy_prefix = pv.global_theme.trame.server_proxy_prefix
        # server_proxy_prefix assumes trailing slash
        prefix = server_proxy_prefix if server_proxy_prefix else ''
        src = f'{prefix}{_server.port}/index.html{params}'
    else:
        src = f'{protocol}://{host}:{_server.port}/index.html{params}'
    logger.debug(src)
    return src


def initialize(
    server,
    plotter,
    mode=None,
    default_server_rendering=True,
    collapse_menu=False,
    **kwargs,
):  # numpydoc ignore=PR01,RT01
    """Generate the UI for a given plotter."""
    state = server.state
    state.trame__title = UI_TITLE

    viewer = get_viewer(
        plotter,
        server=server,
        suppress_rendering=mode == 'client',
    )

    with viewer.make_layout(server, template_name=plotter._id_name) as layout:
        viewer.layout = layout
        viewer.ui(
            mode=mode,
            default_server_rendering=default_server_rendering,
            collapse_menu=collapse_menu,
            **kwargs,
        )

    return viewer


def show_trame(
    plotter: Plotter,
    mode: JupyterBackendOptions | None = None,
    name: str | None = None,
    server_proxy_enabled: bool | None = None,
    server_proxy_prefix: str | None = None,
    jupyter_extension_enabled: bool | None = None,
    collapse_menu: bool = False,
    add_menu: bool = True,
    add_menu_items: Callable[[Literal['trame', 'server', 'client'], bool, bool], None]
    | None = None,
    default_server_rendering: bool = True,
    handler: Callable[Concatenate[Viewer, str, ...], IFrame] | None = None,
    **kwargs,
) -> EmbeddableWidget | IFrame | Widget:
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
        * ``'html'``: Exports the scene for client rendering that can be
          embedded in a webpage.

    name : str, optional
        The name of the trame server on which the UI is defined.

    server_proxy_enabled : bool, default: False
        Build a relative URL for use with ``jupyter-server-proxy``.

    server_proxy_prefix : str, optional
        URL prefix when using ``server_proxy_enabled``. This can be set
        globally in the theme. To ignore, pass ``False``. For use with
        ``jupyter-server-proxy``, often set to ``proxy/``.

    jupyter_extension_enabled : bool, default: False
        Build a relative URL for use with ``trame-jupyter-extension``.

    collapse_menu : bool, default: False
        Collapse the UI menu (camera controls, etc.) on start.

    add_menu : bool, default: True
        Add a UI controls VCard to the VContainer.

    add_menu_items : callable, default: None
        Append more UI controls to the VCard menu. Should be a function similar to
        Viewer.ui_controls().

    default_server_rendering : bool, default: True
        Whether to use server-side or client-side rendering on-start when
        using the ``'trame'`` mode.

    handler : callable, optional
        Pass a callable that accptes the viewer instance, the string URL,
        and ``**kwargs`` to create custom HTML representations of the output.

        .. code-block:: python

            import pyvista as pv
            from IPython.display import IFrame

            mesh = pyvista.Wavelet()


            def handler(viewer, src, **kwargs):
                return IFrame(src, '75%', '500px')


            pl = pv.Plotter(notebook=True)
            _ = pl.add_mesh(mesh)
            iframe = pl.show(
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
    output : ipywidgets.widgets.HTML or handler result
        Returns a HTML IFrame widget or the result of the passed handler.

    """
    if plotter.render_window is None:
        raise RuntimeError(CLOSED_PLOTTER_ERROR)

    if plotter._window_size_unset:
        dw, dh = '99%', '600px'
    else:
        width, height = plotter.window_size
        dw = f'{width}px'
        dh = f'{height}px'
    kwargs.setdefault('width', dw)
    kwargs.setdefault('height', dh)

    if mode == 'html':
        return EmbeddableWidget(plotter, **kwargs)

    if jupyter_extension_enabled is None:
        jupyter_extension_enabled = pv.global_theme.trame.jupyter_extension_enabled

    if name is None:
        server = get_server(name=pv.global_theme.trame.jupyter_server_name)
    else:
        server = get_server(name=name)
    if name is None and not server.running:
        wslink_backend = 'aiohttp'
        if jupyter_extension_enabled:  # pragma: no cover
            wslink_backend = 'jupyter'

        elegantly_launch(server, wslink_backend=wslink_backend)
        if not server.running:  # pragma: no cover
            raise TrameJupyterServerDownError
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
        add_menu_items=add_menu_items,
    )

    if jupyter_extension_enabled:  # pragma: no cover
        from trame_client.ui.core import iframe_url_builder_jupyter_extension  # noqa: PLC0415

        iframe_attrs = iframe_url_builder_jupyter_extension(viewer.layout)
        src = iframe_attrs['src']
    else:
        # TODO: The build_url function could possibly be replaced by
        # trame's upstream url builders in trame_client.ui.core
        iframe_attrs = {}
        src = build_url(
            server,
            ui=plotter._id_name,
            server_proxy_enabled=server_proxy_enabled,
            server_proxy_prefix=server_proxy_prefix,
            host=kwargs.get('host', 'localhost'),
            protocol=kwargs.get('protocol', 'http'),
        )

    if callable(handler):
        return handler(viewer, src, iframe_attrs=iframe_attrs, **kwargs)
    return Widget(viewer, src, iframe_attrs=iframe_attrs, **kwargs)


def elegantly_launch(*args, **kwargs):  # numpydoc ignore=PR01
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
        import nest_asyncio2  # noqa: PLC0415
    except ImportError:
        msg = (
            'Please install `nest_asyncio2` to automagically launch the trame server '
            'without await. Or, to avoid `nest_asyncio2` run:\n\n'
            'from pyvista.trame.jupyter import launch_server\n'
            'await launch_server().ready'
        )
        raise ImportError(msg)

    async def launch_it():
        await launch_server(*args, **kwargs).ready

    # Basically monkey patches asyncio to support this
    nest_asyncio2.apply()

    return asyncio.run(launch_it())
