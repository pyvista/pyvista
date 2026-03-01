from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

import pyvista as pv
from pyvista.trame import PyVistaLocalView

server = get_server()
server.client_type = 'vue2'
state, ctrl = server.state, server.controller

print(pv.Report())

state.trame__title = "PyVista Volume"

# -----------------------------------------------------------------------------

vol = pv.Wavelet()

plotter = pv.Plotter(off_screen=True)
actor = plotter.add_volume(vol, mapper='smart')
plotter.set_background('lightgrey')


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text("PyVistaLocalView")

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
