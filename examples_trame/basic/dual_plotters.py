from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk as vtk_widgets, vuetify

import pyvista as pv

pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "Multi Plotters"


mesh = pv.Wavelet()

pl1 = pv.Plotter()
pl1.add_mesh(mesh.contour())
pl1.view_isometric()

pl2 = pv.Plotter()
pl2.add_mesh(mesh.outline(), color='black')
pl2.view_isometric()


def my_callback(normal, origin):
    pl2.add_mesh(mesh.slice(normal, origin), name="slice")
    # ctrl.view2_update()  # <-- is this being throttled?
    pl2.add_timer_event(
        max_steps=1, duration=1000, callback=ctrl.view2_update
    )


pl1.add_plane_widget(my_callback)


with SinglePageLayout(server) as layout:
    layout.title.set_text("Multi Views")
    layout.icon.click = ctrl.view_reset_camera

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            with vuetify.VCol(classes="pa-0 fill-height"):
                view = vtk_widgets.VtkRemoteView(pl1.render_window, ref="view1")
                ctrl.view1_update = view.update
            with vuetify.VCol(classes="pa-0 fill-height"):
                view = vtk_widgets.VtkRemoteView(pl2.render_window, ref="view2")
                ctrl.view2_update = view.update

server.start()
