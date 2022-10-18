from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout, VAppLayout
from trame.widgets import vuetify, html

from pyvista.trame import PyVistaLocalView, PyVistaRemoteView

UI_TITLE = "PyVista"

def initialize(server, plotter, local_rendering=True):
    state, ctrl = server.state, server.controller
    state.trame__title = UI_TITLE

    with VAppLayout(server) as layout:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            # with vuetify.VCard(style="position: absolute; top: 20px; left: 20px; z-index: 1",):
            #     with vuetify.VCardTitle(classes="py-0"):
            #         with vuetify.VBtn(icon=True, click="show_content=!show_content"):
            #             vuetify.VIcon('mdi-menu')
            #     with vuetify.VCardText(v_show=("show_content", False)):
            #         html.Div("!!")
            if local_rendering:
                view = PyVistaLocalView(plotter)
            else:
                view = PyVistaRemoteView(plotter)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
