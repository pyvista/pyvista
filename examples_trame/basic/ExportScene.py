"""Export scene for offline rendering.

https://kitware.github.io/vtk-js/examples/OfflineLocalView.html
"""
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify

import pyvista as pv
from pyvista import examples
from pyvista.trame import PyVistaLocalView

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "PyVistaLocalView Export"

mesh = examples.load_random_hills()

plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(mesh)
plotter.set_background('lightgrey')


@ctrl.trigger("export")
def export_scene():
    data = ctrl.view_export(format="zip")
    return server.protocol.addAttachment(memoryview(data))


with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text("Local rendering export")

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VBtn(
            "Export",
            click="utils.download('scene-extract.vtksz', trigger('export'), 'application/octet-stream')",
        )

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            with vuetify.VCol(classes="pa-0 ma-1 fill-height"):
                view = PyVistaLocalView(plotter)
                ctrl.view_export = view.export
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

if __name__ == "__main__":
    server.start()
