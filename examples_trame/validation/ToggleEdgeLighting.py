"""Validate Int64 usage with VTK.js."""

import numpy as np
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify
from trame.widgets.vtk import VtkLocalView, VtkRemoteView

import pyvista as pv

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "Lighting Validation"

# -----------------------------------------------------------------------------

mesh = pv.Wavelet()

plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(mesh)
plotter.reset_camera()


def toggle_edges():
    actor.GetProperty().SetEdgeVisibility(not actor.GetProperty().GetEdgeVisibility())
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(state.trame__title)

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VBtn("Toggle edges", click=toggle_edges)

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            with vuetify.VCol(classes="fill-height"):
                view = VtkLocalView(plotter.ren_win)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera
            with vuetify.VCol(classes="fill-height"):
                VtkRemoteView(plotter.ren_win)

    # hide footer
    layout.footer.hide()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
