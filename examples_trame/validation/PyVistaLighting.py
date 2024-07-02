"""Validate lighting."""

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import html, vuetify
from trame.widgets.vtk import VtkLocalView, VtkRemoteView

import pyvista as pv

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "PyVista Lighting"

# -----------------------------------------------------------------------------

mesh = pv.Cone()

plotter = pv.Plotter(off_screen=True, lighting="light kit")  # ighting='three lights'
actor = plotter.add_mesh(mesh, color='white')
plotter.set_background("paraview")

# light = pv.Light(light_type='headlight')
# light = pv.Light(position=(1, 0, 0), light_type='camera light')
# light = pv.Light(position=(0, 1, 0), light_type='scene light')
# plotter.add_light(light)

plotter.view_isometric()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(state.trame__title)

    with layout.toolbar:
        vuetify.VSpacer()

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
