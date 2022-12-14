# flake8: noqa: D102,D103,D107
"""PyVista Trame User Interface.

This module builds a base UI for manipulating a PyVista Plotter.
The UI generated here is the default for rendering in Jupyter
environments and provides a starting point for custom user-built
applications.
"""
from trame.ui.vuetify import VAppLayout
from trame.widgets import vuetify

import pyvista
from pyvista.plotting.plotting import BasePlotter
from pyvista.trame import PyVistaLocalView, PyVistaRemoteView

UI_TITLE = "PyVista"


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
        self._ctrl = ctrl
        self._state = state

        self.plotter = plotter

        # controller
        ctrl.get_render_window = lambda: self.plotter.ren_win

        # Listen to changes
        self._state.change("view_edge_visiblity")(self.on_edge_visiblity_change)
        self._state.change("grid_visiblity")(self.on_grid_visiblity_change)
        self._state.change("outline_visiblity")(self.on_outline_visiblity_change)

    @vuwrap
    def on_edge_visiblity_change(self, view_edge_visiblity, **kwargs):
        """Toggle edge visibility for all actors."""
        for _, actor in self.plotter.actors.items():
            if isinstance(actor, pyvista.Actor):
                actor.prop.show_edges = view_edge_visiblity

    @vuwrap
    def view_isometric(self):
        """View isometric."""
        self.plotter.view_isometric()

    @vuwrap
    def view_yz(self):
        """View YZ plane."""
        self.plotter.view_yz()

    @vuwrap
    def view_xz(self):
        """View XZ plane."""
        self.plotter.view_xz()

    @vuwrap
    def view_xy(self):
        """View XY plane."""
        self.plotter.view_xy()

    @vuwrap
    def reset_camera(self):
        """Reset the camera."""
        self.plotter.reset_camera()

    @vuwrap
    def on_grid_visiblity_change(self, grid_visiblity, **kwargs):
        """Handle axes grid visibility."""
        if grid_visiblity:
            self.plotter.show_grid()
        else:
            self.plotter.remove_bounds_axes()

    @vuwrap
    def on_outline_visiblity_change(self, outline_visiblity, **kwargs):
        """Handle outline visibility."""
        if outline_visiblity:
            self.plotter.add_bounding_box()
        else:
            self.plotter.remove_bounding_box()

    @property
    def actors(self):
        """Get dataset actors."""
        return {k: v for k, v in self.plotter.actors.items() if isinstance(v, pyvista.Actor)}


def initialize(server, plotter, local_rendering=True):
    """Generate the UI for a given plotter."""
    state, ctrl = server.state, server.controller
    state.trame__title = UI_TITLE

    viewer = Viewer(plotter, server)

    with VAppLayout(server):
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            with vuetify.VCard(
                style="position: absolute; top: 20px; left: 20px; z-index: 1",
            ):
                with vuetify.VCardTitle(classes="py-0"):
                    # Scene controls
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
                        v_model=("grid_visiblity", False),
                        dense=True,
                        hide_details=True,
                        on_icon="mdi-ruler-square",
                        off_icon="mdi-ruler-square",
                        classes="ma-2",
                    )
                    vuetify.VCheckbox(
                        v_model=("outline_visiblity", False),
                        dense=True,
                        hide_details=True,
                        on_icon="mdi-cube",
                        off_icon="mdi-cube-off",
                        classes="ma-2",
                    )

            if local_rendering:
                view = PyVistaLocalView(plotter)
            else:
                view = PyVistaRemoteView(plotter)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
