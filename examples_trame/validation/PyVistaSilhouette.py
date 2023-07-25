from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk as vtk_widgets, vuetify
from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
from vtkmodules.vtkFiltersSources import vtkConeSource

# VTK factory initialization
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

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "VTK Remote View - Local Rendering"

# -----------------------------------------------------------------------------
# VTK pipeline
# -----------------------------------------------------------------------------

DEFAULT_RESOLUTION = 6

renderer = vtkRenderer()
renderWindow = vtkRenderWindow()
renderWindow.AddRenderer(renderer)

renderWindowInteractor = vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

cone_source = vtkConeSource()
mapper = vtkPolyDataMapper()
actor = vtkActor()
mapper.SetInputConnection(cone_source.GetOutputPort())
actor.SetMapper(mapper)
renderer.AddActor(actor)

sil = vtkPolyDataSilhouette()
sil.SetInputConnection(cone_source.GetOutputPort())
sil.SetCamera(renderer.GetActiveCamera())
silmapper = vtkPolyDataMapper()
silmapper.SetInputConnection(sil.GetOutputPort())
silactor = vtkActor()
silactor.GetProperty().SetColor(1, 0, 0)
silactor.GetProperty().SetLineWidth(10)
silactor.SetMapper(silmapper)
renderer.AddActor(silactor)

renderer.ResetCamera()
renderWindow.Render()

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@state.change("resolution")
def update_cone(resolution=DEFAULT_RESOLUTION, **kwargs):
    cone_source.SetResolution(resolution)
    ctrl.view2_update()
    ctrl.view_update()


def update_reset_resolution():
    state.resolution = DEFAULT_RESOLUTION


@ctrl.set("view_on_end_animation")
def on_end_animation(cameraInfo):
    camera = renderer.GetActiveCamera()
    camera.SetPosition(cameraInfo.get("position"))
    camera.SetFocalPoint(cameraInfo.get("focalPoint"))
    camera.SetViewUp(cameraInfo.get("viewUp"))
    camera.SetViewAngle(cameraInfo.get("viewAngle"))
    renderer.ResetCameraClippingRange()

    ctrl.view2_update()
    # mtime2 = sil.GetMTime()
    # sil.Update()
    # cells = sil.GetOutput().GetLines()
    # print(cells)

    # silmapper.Update()
    # renderWindow.Render()
    ctrl.view_update()


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text("Cone Application")

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VSlider(
            v_model=("resolution", DEFAULT_RESOLUTION),
            min=3,
            max=60,
            step=1,
            hide_details=True,
            dense=True,
            style="max-width: 300px",
        )
        vuetify.VDivider(vertical=True, classes="mx-2")
        with vuetify.VBtn(icon=True, click=update_reset_resolution):
            vuetify.VIcon("mdi-undo-variant")

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            with vuetify.VCol(classes="pa-0 fill-height"):
                view = vtk_widgets.VtkLocalView(
                    renderWindow,
                    ref="local",
                    interactor_events=("vtk_events", ["EndAnimation"]),
                    EndAnimation=(
                        ctrl.view_on_end_animation,
                        "[$event.pokedRenderer.getActiveCamera().get()]",
                    ),
                )
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera
            with vuetify.VCol(classes="pa-0 fill-height"):
                vr = vtk_widgets.VtkRemoteView(
                    renderWindow,
                    ref="remote",
                )
                ctrl.view2_update = vr.update


# -----------------------------------------------------------------------------
# Jupyter
# -----------------------------------------------------------------------------


def show(**kwargs):
    from trame.app import jupyter

    jupyter.show(server, **kwargs)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
