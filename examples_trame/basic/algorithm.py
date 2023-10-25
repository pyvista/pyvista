from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

import pyvista as pv
from pyvista.trame.ui import plotter_ui

# -----------------------------------------------------------------------------
# Trame initialization
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "Cone"
ctrl.on_server_ready.add(ctrl.view_update)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

source = pv.ConeSource()

pl = pv.Plotter()
pl.add_mesh(source, color='seagreen')


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@state.change("resolution")
def update_contour(resolution, **kwargs):
    source.resolution = int(resolution)
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    layout.title.set_text("Cone")

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VSlider(
            v_model=("resolution", 15),
            min=5,
            max=30,
            hide_details=True,
            dense=True,
            style="max-width: 300px",
            change=ctrl.view_update,
        )

        vuetify.VProgressLinear(
            indeterminate=True,
            absolute=True,
            bottom=True,
            active=("trame__busy",),
        )

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            # Use PyVista UI template for Plotters
            view = plotter_ui(pl)
            ctrl.view_update = view.update

server.start()
