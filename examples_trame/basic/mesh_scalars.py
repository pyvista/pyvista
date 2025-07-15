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

state.trame__title = 'Modify Mapped Scalars'
ctrl.on_server_ready.add(ctrl.view_update)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

mesh = examples.download_antarctica_velocity()

pl = pv.Plotter()
actor = pl.add_mesh(mesh)
pl.view_xy()


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@state.change('scalars')
def set_scalars(scalars=mesh.active_scalars_name, **kwargs):  # noqa: ARG001
    actor.mapper.array_name = scalars
    actor.mapper.scalar_range = mesh.get_data_range(scalars)
    ctrl.view_update()


@state.change('log_scale')
def set_log_scale(*, log_scale=False, **kwargs):  # noqa: ARG001
    actor.mapper.lookup_table.log_scale = log_scale
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    layout.title.set_text('Scalar Selection')
    layout.icon.click = ctrl.view_reset_camera

    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VCheckbox(
            label='Log Scale',
            v_model=('log_scale', False),
            hide_details=True,
            density='compact',
            outlined=True,
            classes='pt-1 ml-2',
        )
        vuetify3.VSelect(
            label='Scalars',
            v_model=('scalars', mesh.active_scalars_name),
            items=('array_list', list(mesh.point_data.keys())),
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
            view = plotter_ui(pl, default_server_rendering=True)
            ctrl.view_update = view.update

server.start()
