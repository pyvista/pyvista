from __future__ import annotations

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import pyvista as pv
from pyvista.trame.ui import plotter_ui

# -----------------------------------------------------------------------------
# Trame initialization
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'Cone'
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


@state.change('resolution')
def update_contour(resolution, **kwargs):  # noqa: ARG001
    source.resolution = int(resolution)
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    layout.title.set_text('Cone')

    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VSlider(
            label='Resolution',
            v_model=('resolution', 15),
            min=5,
            max=30,
            hide_details=True,
            density='compact',
            style='max-width: 300px',
            change=ctrl.view_update,
        )

        vuetify3.VProgressLinear(
            indeterminate=True,
            absolute=True,
            bottom=True,
            active=('trame__busy',),
        )

    with layout.content:
        with vuetify3.VContainer(
            fluid=True,
            classes='pa-0 fill-height',
        ):
            # Use PyVista UI template for Plotters
            view = plotter_ui(pl)
            ctrl.view_update = view.update

server.start()
