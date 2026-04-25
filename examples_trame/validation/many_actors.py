from __future__ import annotations

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import pyvista as pv
from pyvista.trame.ui import plotter_ui

pv.OFF_SCREEN = True

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'PyVista Many Actors'

# -----------------------------------------------------------------------------

pl = pv.Plotter()

for i in range(25):
    for j in range(25):
        actor = pl.add_mesh(pv.Sphere(center=(i, j, 0)))


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(state.trame__title)

    with layout.toolbar:
        vuetify3.VSpacer()

    with layout.content:
        # Use PyVista UI template for Plotters
        view = plotter_ui(pl)
        ctrl.view_update = view.update

    # hide footer
    layout.footer.hide()

server.start()
