import pyvista as pv
from pyvista import examples

from pyvista.trame import PyVistaLocalView
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify


server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "PyVista Local View"

# -----------------------------------------------------------------------------

mesh = examples.load_random_hills()

plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(mesh)
plotter.view_isometric()
plotter.set_background('lightgrey')


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text("PyVista Local View")

    with layout.toolbar:
        vuetify.VSpacer()

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            view = PyVistaLocalView(plotter)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera

    # hide footer
    layout.footer.hide()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
