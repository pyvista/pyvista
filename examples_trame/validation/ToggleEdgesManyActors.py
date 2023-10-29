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

TITLE = "Local Toggle Edges with Many Actors"

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

actors = []
for i in range(50):
    for j in range(50):
        cone_source = vtkConeSource()
        cone_source.SetCenter((i, j, 0))
        mapper = vtkPolyDataMapper()
        actor = vtkActor()
        mapper.SetInputConnection(cone_source.GetOutputPort())
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        actors.append(actor)
renderer.ResetCamera()
renderWindow.Render()
renderer.SetBackground(0, 0.5, 0)

camera = renderer.GetActiveCamera()

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


def toggle_edges():
    for actor in actors:
        actor.GetProperty().SetEdgeVisibility(not actor.GetProperty().GetEdgeVisibility())
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text(TITLE)

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VBtn("Toggle edges", click=toggle_edges)

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            local_view = vtk_widgets.VtkLocalView(
                renderWindow,
                ref="view_local",
            )
            ctrl.view_update = local_view.update
            ctrl.view_reset_camera = local_view.reset_camera
            ctrl.view_push_camera = local_view.push_camera


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
