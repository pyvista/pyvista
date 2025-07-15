# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
#   "trame>=2.5.2",
# ]
# ///

from __future__ import annotations

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import pyvista as pv
from pyvista import examples
from pyvista.trame.ui import plotter_ui

# -----------------------------------------------------------------------------
# Trame initialization
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'Multi Views'
ctrl.on_server_ready.add(ctrl.view_update)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
mesh = examples.load_random_hills()
arrows = mesh.glyph(scale='Normals', orient='Normals', tolerance=0.05)


pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(mesh, scalars='Elevation', cmap='terrain', smooth_shading=True)
pl.subplot(0, 1)
pl.add_mesh(mesh, opacity=0.75, scalars='Elevation', cmap='terrain', smooth_shading=True)
pl.add_mesh(arrows, color='black')
pl.link_views()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    layout.title.set_text('Multi Views')
    layout.icon.click = ctrl.view_reset_camera

    with layout.content:
        with vuetify3.VContainer(
            fluid=True,
            classes='pa-0 fill-height',
        ):
            # Use PyVista UI template for Plotters
            view = plotter_ui(pl)
            ctrl.view_update = view.update

server.start()
