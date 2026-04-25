"""How to use PyVista's ``PyVistaLocalView`` trame view component.

This is a full-fledged example on building your own user interface
with client-side rendering.
"""

from __future__ import annotations

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import pyvista as pv
from pyvista import examples
from pyvista.trame import PyVistaLocalView

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'PyVistaLocalView'

# -----------------------------------------------------------------------------

mesh = examples.load_random_hills()

pl = pv.Plotter(off_screen=True)
actor = pl.add_mesh(mesh)
pl.set_background('lightgrey')


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text('PyVistaLocalView')

    with layout.toolbar:
        vuetify3.VSpacer()

    with layout.content:
        with vuetify3.VContainer(
            fluid=True,
            classes='pa-0 fill-height',
        ):
            view = PyVistaLocalView(pl)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera

    # hide footer
    layout.footer.hide()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    server.start()
