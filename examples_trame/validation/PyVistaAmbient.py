"""Validate lighting and properties."""

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import html, vuetify
from trame.widgets.vtk import VtkLocalView, VtkRemoteView

import pyvista as pv

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "PyVista Ambient Lighting"

# -----------------------------------------------------------------------------

mesh = pv.Cone()

plotter = pv.Plotter(off_screen=True, lighting="none")
actor = plotter.add_mesh(mesh, ambient=0.5, specular=0.5, specular_power=100, color='lightblue')
plotter.set_background("paraview")
plotter.view_isometric()

# light = pv.Light(light_type='headlight')  # GOOD
# light = pv.Light(position=(1, 0, 0), light_type='camera light')  # BAD
light = pv.Light(position=(0, 1, 0), light_type="scene light")  # BAD
plotter.add_light(light)


@state.change("color")
def color(color='lightblue', **kwargs):
    actor.prop.color = color
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(state.trame__title)

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VSelect(
            label="Color",
            v_model=("color", 'lightblue'),
            items=("array_list", ['lightblue', "#0000ff", "white"]),
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
            with vuetify.VContainer(fluid=True, classes="pa-0 fill-height", style="width: 50%;"):
                local = VtkLocalView(
                    plotter.ren_win,
                )
            with vuetify.VContainer(fluid=True, classes="pa-0 fill-height", style="width: 50%;"):
                remote = VtkRemoteView(
                    plotter.ren_win,
                )

            def view_update(**kwargs):
                local.update(**kwargs)
                remote.update(**kwargs)

            def view_reset_camera(**kwargs):
                local.reset_camera(**kwargs)
                remote.reset_camera(**kwargs)

            ctrl.view_update = view_update
            ctrl.view_reset_camera = view_reset_camera

            ctrl.on_server_ready.add(view_update)

    # hide footer
    layout.footer.hide()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
