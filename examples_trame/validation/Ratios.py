from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify
from trame.widgets.vtk import VtkRemoteView

import pyvista as pv

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "PyVista Remote View Ratios"

# -----------------------------------------------------------------------------

mesh = pv.Wavelet()

plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(mesh)
plotter.set_background("lightgrey")
plotter.show_grid()
plotter.view_isometric()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(state.trame__title)

    with layout.toolbar:
        vuetify.VSpacer()

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            view = VtkRemoteView(plotter.ren_win, interactive_ratio=2, still_ratio=2)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
