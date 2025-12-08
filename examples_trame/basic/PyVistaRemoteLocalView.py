"""How to use PyVista's ``PyVistaRemoteView`` trame view component.

This is a full-fledged example on building your own user interface
with server-side rendering.
"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "matplotlib",
#   "pyvista",
#   "trame>=2.5.2",
# ]
# ///

from __future__ import annotations

import matplotlib.pyplot as plt
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import pyvista as pv
from pyvista import examples
from pyvista.trame import PyVistaRemoteLocalView

pv.OFF_SCREEN = True

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'PyVistaRemoteView'

# -----------------------------------------------------------------------------

mesh = examples.load_random_hills()

pl = pv.Plotter()
actor = pl.add_mesh(mesh, cmap='viridis')


@state.change('cmap')
def update_cmap(cmap='viridis', **kwargs):  # noqa: ARG001
    actor.mapper.lookup_table.cmap = cmap
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text('PyVistaRemoteLocalView')

    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VCheckbox(
            v_model=('use_server_rendering', False),
            density='compact',
            hide_details=True,
            true_icon='mdi-dns',
            false_icon='mdi-open-in-app',
            classes='ma-2',
        )
        vuetify3.VSpacer()
        vuetify3.VSelect(
            label='Color map',
            v_model=('cmap', 'viridis'),
            items=('array_list', plt.colormaps()),
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
            view = PyVistaRemoteLocalView(
                pl,
                mode=("use_server_rendering ? 'remote' : 'local'", 'local'),
            )
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera

    # hide footer
    layout.footer.hide()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    server.start()
