from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify
from trame.widgets.vtk import VtkLocalView, VtkRemoteView
import vtk

server = get_server()
state, ctrl = server.state, server.controller

state.trame__title = "VTK Volume Rendering"

# -----------------------------------------------------------------------------
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

source = vtk.vtkRTAnalyticSource()
source.Update()
# mapper = vtk.vtkFixedPointVolumeRayCastMapper()  # works!
mapper = vtk.vtkSmartVolumeMapper()  # fails!
# mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()  # fails
# mapper = vtk.vtkGPUVolumeRayCastMapper()  # fails
mapper.SetInputConnection(source.GetOutputPort())
actor = vtk.vtkVolume()
actor.SetMapper(mapper)
actor.GetProperty().SetScalarOpacityUnitDistance(10)
ren.AddActor(actor)

colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(20, 0.0)
opacityTransferFunction.AddPoint(255, 0.2)

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationTypeToLinear()

actor.SetProperty(volumeProperty)

cube = vtk.vtkCubeAxesActor()
cube.SetCamera(ren.GetActiveCamera())
cube.SetBounds(source.GetOutput().GetBounds())
ren.AddActor(cube)

iren.Initialize()
ren.ResetCamera()
ren.SetBackground(0.7, 0.7, 0.7)
renWin.Render()


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
                local = VtkLocalView(renWin)
            with vuetify.VContainer(fluid=True, classes="pa-0 fill-height", style="width: 50%;"):
                remote = VtkRemoteView(renWin)

    # hide footer
    layout.footer.hide()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
