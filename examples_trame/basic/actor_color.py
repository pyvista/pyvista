from __future__ import annotations

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import pyvista as pv
from pyvista.plotting.colors import hexcolors
from pyvista.trame.ui import plotter_ui

# -----------------------------------------------------------------------------
# Trame initialization
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'Actor Color'
ctrl.on_server_ready.add(ctrl.view_update)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

mesh = pv.Cone()

pl = pv.Plotter()
actor = pl.add_mesh(mesh, color='seagreen')


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@state.change('color')
def color(color='seagreen', **kwargs):  # noqa: ARG001
    actor.prop.color = color
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    layout.title.set_text('Actor Color')
    layout.icon.click = ctrl.view_reset_camera

    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VSelect(
            label='Color',
            v_model=('color', 'seagreen'),
            items=('array_list', list(hexcolors.keys())),
            hide_details=True,
            density='compact',
            outlined=True,
            classes='pt-1 ml-2',
            style='max-width: 250px',
        )

    with layout.content:
        with vuetify3.VContainer(
            fluid=True,
            classes='pa-0 fill-height',
        ):
            # Use PyVista UI template for Plotters
            view = plotter_ui(pl, default_server_rendering=False)
            ctrl.view_update = view.update

server.start()
