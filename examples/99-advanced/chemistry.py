#!/usr/bin/env python
from vtkmodules.util.misc import vtkGetDataRoot
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOChemistry import vtkGaussianCubeReader
from vtkmodules.vtkImagingCore import vtkImageShiftScale
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)
import vtkmodules.vtkRenderingFreeType
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401

VTK_DATA_ROOT = vtkGetDataRoot()

ren1 = vtkRenderer()
renWin = vtkRenderWindow()
renWin.SetMultiSamples(0)
renWin.AddRenderer(ren1)

renWin.SetSize(300, 300)

iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

camera = vtkCamera()
camera.ParallelProjectionOn()
camera.SetViewUp(0, 1, 0)
camera.SetFocalPoint(12, 10.5, 15)
camera.SetPosition(-70, 15, 34)
camera.ComputeViewPlaneNormal()
ren1.SetActiveCamera(camera)
# Create the reader for the data
# vtkStructuredPointsReader reader
reader = vtkGaussianCubeReader()
reader.SetFileName("m4_TotalDensity.cube")
reader.SetHBScale(1.1)
reader.SetBScale(10)
reader.Update()

range_ = reader.GetGridOutput().GetPointData().GetScalars().GetRange()

min_ = range_[0]
max_ = range_[1]

readerSS = vtkImageShiftScale()
readerSS.SetInputData(reader.GetGridOutput())
readerSS.SetShift(min_ * -1)
readerSS.SetScale(255 / (max_ - min_))
readerSS.SetOutputScalarTypeToUnsignedChar()

bounds = vtkOutlineFilter()
bounds.SetInputData(reader.GetGridOutput())

boundsMapper = vtkPolyDataMapper()
boundsMapper.SetInputConnection(bounds.GetOutputPort())

boundsActor = vtkActor()
boundsActor.SetMapper(boundsMapper)
boundsActor.GetProperty().SetColor(0, 0, 0)

contour = vtkContourFilter()
contour.SetInputData(reader.GetGridOutput())
contour.GenerateValues(5, 0, 0.05)

contourMapper = vtkPolyDataMapper()
contourMapper.SetInputConnection(contour.GetOutputPort())
contourMapper.SetScalarRange(0, 0.1)
contourMapper.GetLookupTable().SetHueRange(0.32, 0)

contourActor = vtkActor()
contourActor.SetMapper(contourMapper)
contourActor.GetProperty().SetOpacity(0.5)

# Create transfer mapping scalar value to opacity
opacityTransferFunction = vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(0, 0.01)
opacityTransferFunction.AddPoint(255, 0.35)
opacityTransferFunction.ClampingOn()

# Create transfer mapping scalar value to color
colorTransferFunction = vtkColorTransferFunction()
colorTransferFunction.AddHSVPoint(0.0, 0.66, 1.0, 1.0)
colorTransferFunction.AddHSVPoint(50.0, 0.33, 1.0, 1.0)
colorTransferFunction.AddHSVPoint(100.0, 0.00, 1.0, 1.0)

# The property describes how the data will look
volumeProperty = vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.SetInterpolationTypeToLinear()

# The mapper knows how to render the data
volumeMapper = vtkFixedPointVolumeRayCastMapper()
volumeMapper.SetInputConnection(readerSS.GetOutputPort())

# The volume holds the mapper and the property and
# can be used to position/orient the volume
volume = vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

ren1.AddVolume(volume)

# ren1 AddActor contourActor
ren1.AddActor(boundsActor)

######################################################################
Sphere = vtkSphereSource()
Sphere.SetCenter(0, 0, 0)
Sphere.SetRadius(1)
Sphere.SetThetaResolution(16)
Sphere.SetStartTheta(0)
Sphere.SetEndTheta(360)
Sphere.SetPhiResolution(16)
Sphere.SetStartPhi(0)
Sphere.SetEndPhi(180)

Glyph = vtkGlyph3D()
Glyph.SetInputConnection(reader.GetOutputPort())
Glyph.SetOrient(1)
Glyph.SetColorMode(1)
# Glyph.ScalingOn()
Glyph.SetScaleMode(2)
Glyph.SetScaleFactor(0.6)
Glyph.SetSourceConnection(Sphere.GetOutputPort())

AtomsMapper = vtkPolyDataMapper()
AtomsMapper.SetInputConnection(Glyph.GetOutputPort())
AtomsMapper.UseLookupTableScalarRangeOff()
AtomsMapper.SetScalarVisibility(1)
AtomsMapper.SetScalarModeToDefault()

Atoms = vtkActor()
Atoms.SetMapper(AtomsMapper)
Atoms.GetProperty().SetRepresentationToSurface()
Atoms.GetProperty().SetInterpolationToGouraud()
Atoms.GetProperty().SetAmbient(0.15)
Atoms.GetProperty().SetDiffuse(0.85)
Atoms.GetProperty().SetSpecular(0.1)
Atoms.GetProperty().SetSpecularPower(100)
Atoms.GetProperty().SetSpecularColor(1, 1, 1)
Atoms.GetProperty().SetColor(1, 1, 1)

Tube = vtkTubeFilter()
Tube.SetInputConnection(reader.GetOutputPort())
Tube.SetNumberOfSides(16)
Tube.SetCapping(0)
Tube.SetRadius(0.2)
Tube.SetVaryRadius(0)
Tube.SetRadiusFactor(10)

BondsMapper = vtkPolyDataMapper()
BondsMapper.SetInputConnection(Tube.GetOutputPort())
BondsMapper.UseLookupTableScalarRangeOff()
BondsMapper.SetScalarVisibility(1)
BondsMapper.SetScalarModeToDefault()

Bonds = vtkActor()
Bonds.SetMapper(BondsMapper)
Bonds.GetProperty().SetRepresentationToSurface()
Bonds.GetProperty().SetInterpolationToGouraud()
Bonds.GetProperty().SetAmbient(0.15)
Bonds.GetProperty().SetDiffuse(0.85)
Bonds.GetProperty().SetSpecular(0.1)
Bonds.GetProperty().SetSpecularPower(100)
Bonds.GetProperty().SetSpecularColor(1, 1, 1)
Bonds.GetProperty().SetColor(1, 1, 1)
ren1.AddActor(Bonds)
ren1.AddActor(Atoms)
####################################################
ren1.SetBackground(1, 1, 1)
ren1.ResetCamera()

renWin.Render()


def TkCheckAbort(obj=None, event=""):
    if renWin.GetEventPending():
        renWin.SetAbortRender(1)


renWin.AddObserver("AbortCheckEvent", TkCheckAbort)

iren.Initialize()
iren.Start()
