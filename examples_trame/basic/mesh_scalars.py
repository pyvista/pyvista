from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

import pyvista as pv
from pyvista import examples
from pyvista.trame.ui import plotter_ui

# -----------------------------------------------------------------------------
# Trame initialization
# -----------------------------------------------------------------------------

pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "Modify Mapped Scalars"
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


@state.change("scalars")
def set_scalars(scalars=mesh.active_scalars_name, **kwargs):
    actor.mapper.array_name = scalars
    actor.mapper.scalar_range = mesh.get_data_range(scalars)
    ctrl.view_update()


@state.change("log_scale")
def set_log_scale(log_scale=False, **kwargs):
    actor.mapper.lookup_table.log_scale = log_scale
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


with SinglePageLayout(server) as layout:
    layout.title.set_text("Scalar Selection")
    layout.icon.click = ctrl.view_reset_camera

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VCheckbox(
            label="Log Scale",
            v_model=("log_scale", False),
            hide_details=True,
            dense=True,
            outlined=True,
            # classes="pt-1 ml-2",
        )
        vuetify.VSelect(
            label="Scalars",
            v_model=("scalars", mesh.active_scalars_name),
            items=("array_list", list(mesh.point_data.keys())),
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1 ml-2",
            style="max-width: 250px",
        )

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            # Use PyVista UI template for Plotters
            view = plotter_ui(pl, default_server_rendering=True)
            ctrl.view_update = view.update

server.start()
