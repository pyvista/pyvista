from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

import pyvista as pv
from pyvista.trame.ui import plotter_ui

pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "PyVista Many Actors"

# -----------------------------------------------------------------------------

plotter = pv.Plotter()

for i in range(25):
    for j in range(25):
        actor = plotter.add_mesh(pv.Sphere(center=(i, j, 0)))


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(state.trame__title)

    with layout.toolbar:
        vuetify.VSpacer()

    with layout.content:
        # Use PyVista UI template for Plotters
        view = plotter_ui(plotter)
        ctrl.view_update = view.update

    # hide footer
    layout.footer.hide()

server.start()
