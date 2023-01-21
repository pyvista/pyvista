# flake8: noqa: D102,D103,D107
"""PyVista Trame User Interface.

This module builds a base UI for manipulating a PyVista Plotter.
The UI generated here is the default for rendering in Jupyter
environments and provides a starting point for custom user-built
applications.
"""
import io

from trame.ui.vuetify import VAppLayout
from trame.widgets import html, vuetify

import pyvista
from pyvista.trame.views import PyVistaLocalView, PyVistaRemoteLocalView, PyVistaRemoteView

UI_TITLE = 'PyVista'

VALID_UI_MODES = [
    'trame',
    'client',
    'server',
]


def vuwrap(func):
    """Call view_update in trame to synchronize changes to a view."""

    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self._ctrl.view_update()
        return ret

    return wrapper


class Viewer:
    """Internal wrapper to sync trame view with Plotter."""

    def __init__(self, plotter, server, suppress_rendering=False):
        """Initialize Viewer."""
        state, ctrl = server.state, server.controller
        self._server = server
        self._ctrl = ctrl
        self._state = state

        self.plotter = plotter
        self.plotter.suppress_rendering = suppress_rendering

        # State variable names
        self.SHOW_UI = f'{plotter._id_name}_show_ui'
        self.GRID = f'{plotter._id_name}_grid_visibility'
        self.OUTLINE = f'{plotter._id_name}_outline_visibility'
        self.EDGES = f'{plotter._id_name}_edge_visibility'
        self.AXIS = f'{plotter._id_name}_axis_visiblity'
        self.SCREENSHOT = f'{plotter._id_name}_download_screenshot'
        self.SERVER_RENDERING = f'{plotter._id_name}_use_server_rendering'

        # controller
        ctrl.get_render_window = lambda: self.plotter.ren_win

        # Listen to state changes
        self._state.change(self.EDGES)(self.on_edge_visiblity_change)
        self._state.change(self.GRID)(self.on_grid_visiblity_change)
        self._state.change(self.OUTLINE)(self.on_outline_visiblity_change)
        self._state.change(self.AXIS)(self.on_axis_visiblity_change)
        self._state.change(self.SERVER_RENDERING)(self.on_rendering_mode_change)
        # Listen to events
        self._ctrl.trigger(self.SCREENSHOT)(self.screenshot)

    @vuwrap
    def on_edge_visiblity_change(self, **kwargs):
        """Toggle edge visibility for all actors."""
        value = self._state[self.GRID]
        for _, actor in self.plotter.actors.items():
            if isinstance(actor, pyvista.Actor):
                actor.prop.show_edges = value

    @vuwrap
    def view_isometric(self):
        """View isometric."""
        self.plotter.view_isometric()
        self._ctrl.view_push_camera()

    @vuwrap
    def view_yz(self):
        """View YZ plane."""
        self.plotter.view_yz()
        self._ctrl.view_push_camera()

    @vuwrap
    def view_xz(self):
        """View XZ plane."""
        self.plotter.view_xz()
        self._ctrl.view_push_camera()

    @vuwrap
    def view_xy(self):
        """View XY plane."""
        self.plotter.view_xy()
        self._ctrl.view_push_camera()

    @vuwrap
    def reset_camera(self):
        """Reset the camera."""
        # self.plotter.reset_camera()
        self._ctrl.view_reset_camera()

    @vuwrap
    def on_grid_visiblity_change(self, **kwargs):
        """Handle axes grid visibility."""
        if self._state[self.GRID]:
            self.plotter.show_grid()
        else:
            self.plotter.remove_bounds_axes()

    @vuwrap
    def on_outline_visiblity_change(self, **kwargs):
        """Handle outline visibility."""
        if self._state[self.OUTLINE]:
            self.plotter.add_bounding_box(reset_camera=False)
        else:
            self.plotter.remove_bounding_box()

    @vuwrap
    def on_axis_visiblity_change(self, **kwargs):
        """Handle outline visibility."""
        if self._state[self.AXIS]:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()

    @vuwrap
    def on_rendering_mode_change(self, **kwargs):
        """Handle any configurations when the render mode changes between client and server."""
        if not self._state[self.SERVER_RENDERING]:
            self._ctrl.view_push_camera(force=True)

    @property
    def actors(self):
        """Get dataset actors."""
        return {k: v for k, v in self.plotter.actors.items() if isinstance(v, pyvista.Actor)}

    @vuwrap
    def screenshot(self):
        """Take screenshot and add attachament."""
        self.plotter.render()
        buffer = io.BytesIO()
        self.plotter.screenshot(filename=buffer)
        buffer.seek(0)
        return self._server.protocol.addAttachment(memoryview(buffer.read()))


def button(click, icon, tooltip):
    """Create a vuetify button."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator='{ on, attrs }'):
            with vuetify.VBtn(icon=True, v_bind='attrs', v_on='on', click=click):
                vuetify.VIcon(icon)
        html.Span(tooltip)


def checkbox(model, icons, tooltip):
    """Create a vuetify checkbox."""
    with vuetify.VTooltip(bottom=True):
        with vuetify.Template(v_slot_activator='{ on, attrs }'):
            with html.Div(v_on='on', v_bind='attrs'):
                vuetify.VCheckbox(
                    v_model=model,
                    on_icon=icons[0],
                    off_icon=icons[1],
                    dense=True,
                    hide_details=True,
                    classes='my-0 py-0 ml-1',
                )
        html.Span(tooltip)


def ui_container(
    server, plotter, mode=None, default_server_rendering=True, collapse_menu=False, **kwargs
):
    """Generate VContainer for PyVista Plotter.

    Parameters
    ----------
    server : Server
        The trame server.

    plotter : pyvista.BasePlotter
        The PyVista plotter to connect with the UI.

    mode : str, default: 'trame'
        The UI view mode. Options are:
            * ``'trame'``: Uses a view that can switch between client and server
              rendering modes.
            * ``'server'``: Uses a view that is purely server rendering.
            * ``'client'``: Uses a view that is purely client rendering (generally
              safe without a virtual frame buffer)

    default_server_rendering : bool, default: True
        Whether to use server-side or client-side rendering on-start when
        using the ``'trame'`` mode.

    collapse_menu : bool, default: False
        Collapse the UI menu (camera controls, etc.) on start.

    **kwargs
        Addition keyword arguments are passed to the view being created.

    """
    if mode is None:
        mode = pyvista.global_theme.trame.default_mode
    if mode not in VALID_UI_MODES:
        raise ValueError(f'`{mode}` is not a valid mode choice. Use one of: {VALID_UI_MODES}')
    if mode != 'trame':
        default_server_rendering = mode == 'server'

    viewer = Viewer(plotter, server, suppress_rendering=mode == 'client')
    ctrl = server.controller

    with vuetify.VContainer(
        fluid=True,
        classes='pa-0 fill-height',
    ):
        with vuetify.VCard(
            style='position: absolute; top: 20px; left: 20px; z-index: 1; height: 36px;',
            classes=(f"{{ 'rounded-circle': !{viewer.SHOW_UI} }}",),
        ):
            with vuetify.VRow(classes='pa-0 ma-0'):
                button(
                    click=f'{viewer.SHOW_UI}=!{viewer.SHOW_UI}',
                    icon='mdi-dots-vertical',
                    tooltip=f"{{{{ {viewer.SHOW_UI} ? 'Hide' : 'Show' }}}} menu",
                )
                with vuetify.VRow(
                    v_show=(f'{viewer.SHOW_UI}', not collapse_menu),
                    classes='pa-0 ma-0 align-center',
                ):
                    vuetify.VDivider(vertical=True, classes='mr-1')
                    button(
                        click=viewer.reset_camera,
                        icon='mdi-arrow-expand-all',
                        tooltip='Reset Camera',
                    )
                    vuetify.VDivider(vertical=True, classes='mx-1')
                    button(
                        click=viewer.view_isometric,
                        icon='mdi-axis-arrow',
                        tooltip='Perspective view',
                    )
                    button(
                        click=viewer.view_yz,
                        icon='mdi-axis-x-arrow',
                        tooltip='Reset Camera X',
                    )
                    button(
                        click=viewer.view_xz,
                        icon='mdi-axis-y-arrow',
                        tooltip='Reset Camera Y',
                    )
                    button(
                        click=viewer.view_xy,
                        icon='mdi-axis-z-arrow',
                        tooltip='Reset Camera Z',
                    )
                    vuetify.VDivider(vertical=True, classes='mx-1')
                    checkbox(
                        model=(viewer.OUTLINE, False),
                        icons=('mdi-cube', 'mdi-cube-off'),
                        tooltip=f"Toggle bounding box ({{{{ {viewer.OUTLINE} ? 'on' : 'off' }}}})",
                    )
                    checkbox(
                        model=(viewer.GRID, False),
                        icons=('mdi-ruler-square', 'mdi-ruler-square'),
                        tooltip=f"Toggle ruler ({{{{ {viewer.GRID} ? 'on' : 'off' }}}})",
                    )
                    # Server rendering options
                    if mode == 'trame':
                        vuetify.VDivider(vertical=True, classes='mx-1')
                        checkbox(
                            model=(viewer.SERVER_RENDERING, default_server_rendering),
                            icons=('mdi-dns', 'mdi-open-in-app'),
                            tooltip=f"Toggle rendering mode ({{{{ {viewer.SERVER_RENDERING} ? 'remote' : 'local' }}}})",
                        )
                    with vuetify.VRow(
                        v_show=(viewer.SERVER_RENDERING, default_server_rendering),
                        classes='pa-0 ma-0 align-center',
                    ):
                        checkbox(
                            model=(viewer.AXIS, False),
                            icons=('mdi-axis-arrow-info', 'mdi-axis-arrow-info'),
                            tooltip=f"Toggle axis ({{{{ {viewer.AXIS} ? 'on' : 'off' }}}})",
                        )
                        button(
                            # Must use single-quote string for JS here
                            click=f"utils.download('screenshot.png', trigger('{viewer.SCREENSHOT}'), 'image/png')",
                            icon='mdi-file-png-box',
                            tooltip='Save screenshot',
                        )
        if mode == 'trame':
            view = PyVistaRemoteLocalView(
                plotter,
                mode=(
                    # Must use single-quote string for JS here
                    f"{viewer.SERVER_RENDERING} ? 'remote' : 'local'",
                    'remote' if default_server_rendering else 'local',
                ),
                **kwargs,
            )
            ctrl.view_update_image = view.update_image
        elif mode == 'server':
            view = PyVistaRemoteView(plotter, **kwargs)
        elif mode == 'client':
            view = PyVistaLocalView(plotter, **kwargs)
        ctrl.view_update = view.update
        ctrl.view_reset_camera = view.reset_camera
        ctrl.view_push_camera = view.push_camera

    return plotter._id_name


def initialize(server, plotter, mode=None, default_server_rendering=True, collapse_menu=False):
    """Generate the UI for a given plotter."""
    state = server.state
    state.trame__title = UI_TITLE

    with VAppLayout(server, template_name=plotter._id_name):
        ui_container(
            server,
            plotter,
            mode=mode,
            default_server_rendering=default_server_rendering,
            collapse_menu=collapse_menu,
        )

    # Returns the UI identifier (used in `template_name`)
    return plotter._id_name
