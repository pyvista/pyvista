# flake8: noqa: D102,D103,D107
"""PyVista Trame User Interface.

This module builds a base UI for manipulating a PyVista Plotter.
The UI generated here is the default for rendering in Jupyter
environments and provides a starting point for custom user-built
applications.
"""
import io

from trame.ui.vuetify import VAppLayout
from trame.widgets import vuetify

import pyvista
from pyvista.trame.views import PyVistaLocalView, PyVistaRemoteView

UI_TITLE = 'PyVista'


def vuwrap(func):
    """Call view_update in trame to synchronize changes to a view."""

    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self._ctrl.view_update()
        return ret

    return wrapper


class Viewer:
    """Internal wrapper to sync trame view with Plotter."""

    def __init__(self, plotter, server):
        """Initialize Viewer."""
        state, ctrl = server.state, server.controller
        self._server = server
        self._ctrl = ctrl
        self._state = state

        self.plotter = plotter

        # State variable names
        self.GRID = f'{plotter._id_name}_grid_visibility'
        self.OUTLINE = f'{plotter._id_name}_outline_visibility'
        self.EDGES = f'{plotter._id_name}_edge_visibility'
        self.AXIS = f'{plotter._id_name}_axis_visiblity'
        self.SCREENSHOT = f'{plotter._id_name}_download_screenshot'

        # controller
        ctrl.get_render_window = lambda: self.plotter.ren_win

        # Listen to changes
        self._state.change(self.EDGES)(self.on_edge_visiblity_change)
        self._state.change(self.GRID)(self.on_grid_visiblity_change)
        self._state.change(self.OUTLINE)(self.on_outline_visiblity_change)
        self._state.change(self.AXIS)(self.on_axis_visiblity_change)
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


def ui_container(server, plotter, local_rendering=False):
    """Generate VContainer for PyVista Plotter."""
    ctrl = server.controller

    viewer = Viewer(plotter, server)

    with vuetify.VContainer(
        fluid=True,
        classes='pa-0 fill-height',
    ):
        with vuetify.VCard(
            style='position: absolute; top: 20px; left: 20px; z-index: 1',
        ):
            with vuetify.VCardTitle(classes='py-0'):
                # Scene controls
                with vuetify.VBtn(
                    icon=True, click=f'{plotter._id_name}_show_ui=!{plotter._id_name}_show_ui'
                ):
                    vuetify.VIcon('mdi-dots-vertical')
                with vuetify.VRow(v_show=(f'{plotter._id_name}_show_ui', False)):
                    with vuetify.VBtn(icon=True, click=viewer.view_isometric):
                        vuetify.VIcon('mdi-axis-arrow')
                    with vuetify.VBtn(icon=True, click=viewer.view_yz):
                        vuetify.VIcon('mdi-axis-x-arrow')
                    with vuetify.VBtn(icon=True, click=viewer.view_xz):
                        vuetify.VIcon('mdi-axis-y-arrow')
                    with vuetify.VBtn(icon=True, click=viewer.view_xy):
                        vuetify.VIcon('mdi-axis-z-arrow')
                    with vuetify.VBtn(icon=True, click=viewer.reset_camera):
                        vuetify.VIcon('mdi-arrow-expand-all')
                    vuetify.VCheckbox(
                        v_model=(viewer.OUTLINE, False),
                        dense=True,
                        hide_details=True,
                        on_icon='mdi-cube',
                        off_icon='mdi-cube-off',
                        classes='ma-2',
                    )
                    if not local_rendering:
                        vuetify.VCheckbox(
                            v_model=(viewer.GRID, False),
                            dense=True,
                            hide_details=True,
                            on_icon='mdi-ruler-square',
                            off_icon='mdi-ruler-square',
                            classes='ma-2',
                        )
                        vuetify.VCheckbox(
                            v_model=(viewer.AXIS, False),
                            dense=True,
                            hide_details=True,
                            on_icon='mdi-axis-arrow-info',
                            off_icon='mdi-axis-arrow-info',
                            classes='ma-2',
                        )
                        with vuetify.VBtn(
                            icon=True,
                            click=f"utils.download('screenshot.png', trigger('{viewer.SCREENSHOT}'), 'image/png')",
                        ):
                            vuetify.VIcon('mdi-file-png-box')
        if local_rendering:
            view = PyVistaLocalView(plotter)
            ctrl.view_push_camera = view.push_camera
        else:
            view = PyVistaRemoteView(plotter)
            ctrl.view_push_camera = lambda *args: None
        ctrl.view_update = view.update
        ctrl.view_reset_camera = view.reset_camera

    return plotter._id_name


def initialize(server, plotter, local_rendering=False):
    """Generate the UI for a given plotter."""
    state = server.state
    state.trame__title = UI_TITLE

    with VAppLayout(server, template_name=plotter._id_name):
        ui_container(server, plotter, local_rendering=local_rendering)

    # Returns the UI identifier (used in `template_name`)
    return plotter._id_name
