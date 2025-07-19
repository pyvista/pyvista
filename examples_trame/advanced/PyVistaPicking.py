import numpy as np
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify
from trame.widgets.vtk import VtkLocalView, VtkRemoteView

import pyvista as pv

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "Picking"

# -----------------------------------------------------------------------------

mesh = pv.Wavelet()

plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(mesh)
plotter.reset_camera()
plotter.enable_cell_picking()


def toggle_edges():
    actor.GetProperty().SetEdgeVisibility(not actor.GetProperty().GetEdgeVisibility())
    ctrl.view_update()


@ctrl.set("on_selection_change")
def on_box_selection(event):
    ...


@state.change("selection_mode")
def on_mode_change(selection_mode, **kwargs):
    # Use box for selection
    state.box_selection = selection_mode in [
        "select_surface_point",
        "select_surface_cell",
        "select_frustrum_points",
        "select_frustrum_cells",
        "select_block",
    ]

    # Toggle from interactive to selection
    if selection_mode is None:
        # view.InteractionMode = "3D"
        ...
    else:
        # view.InteractionMode = "Selection"
        ...

    # Handle hover with live update
    state.send_mouse = selection_mode in [
        "select_block",
        "select_hover_point",
        "select_hover_cell",
    ]
    # if state.send_mouse:
    #     asynchronous.create_task(animate())

    # Disable active mode
    if selection_mode == "-":
        state.selection_mode = None


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(state.trame__title)

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VBtn("Toggle edges", click=toggle_edges)

        with vuetify.VBtnToggle(v_model=("selection_mode", None)):
            for name, value in ASSETS.assets.items():
                with vuetify.VBtn(value=name, small=True):
                    vuetify.VImg(
                        src=value,
                        contain=True,
                        height=ICON_SIZE,
                        width=ICON_SIZE,
                    )

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            # with vuetify.VCol(classes="fill-height"):
            #     view = VtkLocalView(plotter.ren_win)
            #     ctrl.view_update = view.update
            #     ctrl.view_reset_camera = view.reset_camera
            # with vuetify.VCol(classes="fill-height"):
            VtkRemoteView(
                plotter.ren_win,
                enable_picking=("send_mouse", False),
                box_selection=("box_selection", False),
                box_selection_change=(ctrl.on_selection_change, "[$event]"),
            )

    # hide footer
    layout.footer.hide()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
