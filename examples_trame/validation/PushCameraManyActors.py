# for remote view
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import html, vtk as vtk_widgets, vuetify
from vtkmodules.vtkFiltersSources import vtkConeSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
import vtkmodules.vtkRenderingOpenGL2  # noqa

# -----------------------------------------------------------------------------
# Trame initialization
# -----------------------------------------------------------------------------

TITLE = "Remote/Local camera sync - many actors"

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = TITLE

# -----------------------------------------------------------------------------
# VTK pipeline
# -----------------------------------------------------------------------------


renderer = vtkRenderer()
renderWindow = vtkRenderWindow()
renderWindow.AddRenderer(renderer)

renderWindowInteractor = vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()


for i in range(50):
    for j in range(50):
        cone_source = vtkConeSource()
        cone_source.SetCenter((i, j, 0))
        mapper = vtkPolyDataMapper()
        actor = vtkActor()
        mapper.SetInputConnection(cone_source.GetOutputPort())
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
renderer.ResetCamera()
renderWindow.Render()

camera = renderer.GetActiveCamera()

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


def push_camera():
    print("Push camera")
    ctrl.view_push_camera()


def push_position():
    print("Push position")
    camera.SetPosition((28.670305828057806, 11.922080695941558, -2.988390267639935))
    ctrl.view_push_camera()
    # ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(TITLE)

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VBtn("Push camera", click=push_camera)
        vuetify.VBtn("Push position", click=push_position)

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
            style="display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr;",
        ):
            with html.Div(
                style="height: 100%;justify-self: stretch;",
            ):
                remote_view = vtk_widgets.VtkRemoteView(
                    renderWindow,
                    ref="view_remote",
                )
                ctrl.view_update.add(remote_view.update)
                ctrl.view_reset_camera.add(remote_view.reset_camera)

            with html.Div(
                style="height: 100%;justify-self: stretch;",
            ):
                local_view = vtk_widgets.VtkLocalView(
                    renderWindow,
                    ref="view_local",
                )
                ctrl.view_update.add(local_view.update)
                ctrl.view_reset_camera.add(local_view.reset_camera)
                ctrl.view_push_camera = local_view.push_camera


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
