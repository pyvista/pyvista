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
from vtkmodules.vtkFiltersCore import vtkContourFilter

import pyvista as pv
from pyvista import examples
from pyvista.trame.ui import plotter_ui

# -----------------------------------------------------------------------------
# Trame initialization
# -----------------------------------------------------------------------------


pv.OFF_SCREEN = True

server = get_server(client_type='vue3')
state, ctrl = server.state, server.controller

state.trame__title = 'Contour'
ctrl.on_server_ready.add(ctrl.view_update)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


volume = examples.download_head_2()

contour = vtkContourFilter()
contour.SetInputDataObject(volume)
# contour.SetComputeNormals(True)
# contour.SetComputeScalars(False)

# Extract data range => Update store/state
data_range = tuple(volume.get_data_range())
contour_value = 0.5 * (data_range[0] + data_range[1])
state.contour_value = contour_value
state.data_range = (float(data_range[0]), float(data_range[1]))

# Configure contour with valid values
contour.SetNumberOfContours(1)
contour.SetValue(0, contour_value)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


pl = pv.Plotter()
actor = pl.add_mesh(contour, cmap='viridis', clim=data_range)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@state.change('contour_value')
def update_contour(contour_value, **kwargs):  # noqa: ARG001
    contour.SetValue(0, contour_value)
    ctrl.view_update_image()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    layout.title.set_text('Contour')

    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VSlider(
            v_model='contour_value',
            min=('data_range[0]',),
            max=('data_range[1]',),
            hide_details=True,
            density='compact',
            style='max-width: 300px',
            start="trigger('demoAnimateStart')",
            end="trigger('demoAnimateStop')",
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
            view = plotter_ui(pl, namespace='demo')
            ctrl.view_update = view.update
            ctrl.view_update_image = view.update_image

server.start()
