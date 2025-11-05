"""How to use PyVista UI template.

This example demonstrates how to use ``plotter_ui`` to add a PyVista
``Plotter`` to a UI with scene controls and standard UI features.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3

import pyvista as pv
from pyvista import examples
from pyvista.trame.ui import get_viewer

pv.OFF_SCREEN = True

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'PyVista UI Template'

# -----------------------------------------------------------------------------

mesh = examples.load_random_hills()

plotter = pv.Plotter()
actor = plotter.add_mesh(mesh, cmap='viridis')


@state.change('cmap')
def update_cmap(cmap='viridis', **kwargs):  # noqa: ARG001
    actor.mapper.lookup_table.cmap = cmap
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

viewer = get_viewer(plotter)

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text('PyVista Colormaps')

    with layout.toolbar:
        # Make sure `mode` matches
        viewer.ui_controls(mode='trame')

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
        # Use PyVista UI template for Plotters
        view = viewer.ui(add_menu=False, mode='trame')
        ctrl.view_update = view.update

    # hide footer
    layout.footer.hide()

server.start()
