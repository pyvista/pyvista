#!/usr/bin/env python
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkImagingCore import vtkImageShiftScale
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkRenderingCore import (
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

import pyvista as pv
from pyvista import examples

ren1 = vtkRenderer()
renWin = vtkRenderWindow()
renWin.SetMultiSamples(0)
renWin.AddRenderer(ren1)

renWin.SetSize(300, 300)

iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

camera = pv.Camera()
camera.ParallelProjectionOn()
camera.SetViewUp(0, 1, 0)
camera.SetFocalPoint(12, 10.5, 15)
camera.SetPosition(-70, 15, 34)
camera.ComputeViewPlaneNormal()
ren1.SetActiveCamera(camera)
# Create the reader for the data
# vtkStructuredPointsReader reader
filename = examples.download_m4_total_density(load=False)
reader = pv.get_reader(filename)
reader.reader.SetHBScale(1.1)
reader.reader.SetBScale(10)
reader.reader.Update()

range_ = reader.reader.GetGridOutput().GetPointData().GetScalars().GetRange()

min_ = range_[0]
max_ = range_[1]

readerSS = vtkImageShiftScale()
readerSS.SetInputData(reader.reader.GetGridOutput())
readerSS.SetShift(min_ * -1)
readerSS.SetScale(255 / (max_ - min_))
readerSS.SetOutputScalarTypeToUnsignedChar()

bounds = vtkOutlineFilter()
bounds.SetInputData(reader.reader.GetGridOutput())

boundsMapper = vtkPolyDataMapper()
boundsMapper.SetInputConnection(bounds.GetOutputPort())

boundsActor = pv.Actor()
boundsActor.SetMapper(boundsMapper)
boundsActor.GetProperty().SetColor(0, 0, 0)

contour = vtkContourFilter()
contour.SetInputData(reader.reader.GetGridOutput())
contour.GenerateValues(5, 0, 0.05)

contourMapper = vtkPolyDataMapper()
contourMapper.SetInputConnection(contour.GetOutputPort())
contourMapper.SetScalarRange(0, 0.1)
contourMapper.GetLookupTable().SetHueRange(0.32, 0)

contourActor = pv.Actor()
contourActor.SetMapper(contourMapper)
contourActor.GetProperty().SetOpacity(0.5)

# Create transfer mapping scalar value to opacity
lut = pv.LookupTable()
opacity_transfer_funtion = lut.to_opacity_tf()
opacity_transfer_funtion.RemoveAllPoints()
opacity_transfer_funtion.AddPoint(0, 0.01)
opacity_transfer_funtion.AddPoint(255, 0.35)
opacity_transfer_funtion.ClampingOn()

# Create transfer mapping scalar value to color
colorTransferFunction = vtkColorTransferFunction()
colorTransferFunction.AddHSVPoint(0.0, 0.66, 1.0, 1.0)
colorTransferFunction.AddHSVPoint(50.0, 0.33, 1.0, 1.0)
colorTransferFunction.AddHSVPoint(100.0, 0.00, 1.0, 1.0)

# The property describes how the data will look
volumeProperty = vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacity_transfer_funtion)
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
sphere = pv.SphereSource()
sphere.SetCenter(0, 0, 0)
sphere.SetRadius(1)
sphere.SetThetaResolution(16)
sphere.SetStartTheta(0)
sphere.SetEndTheta(360)
sphere.SetPhiResolution(16)
sphere.SetStartPhi(0)
sphere.SetEndPhi(180)

glyph = vtkGlyph3D()
glyph.SetInputConnection(reader.reader.GetOutputPort())
glyph.SetOrient(1)
glyph.SetColorMode(1)
# glyph.ScalingOn()
glyph.SetScaleMode(2)
glyph.SetScaleFactor(0.6)
glyph.SetSourceConnection(sphere.GetOutputPort())

atoms_mapper = vtkPolyDataMapper()
atoms_mapper.SetInputConnection(glyph.GetOutputPort())
atoms_mapper.UseLookupTableScalarRangeOff()
atoms_mapper.SetScalarVisibility(1)
atoms_mapper.SetScalarModeToDefault()

atoms = pv.Actor()
atoms.SetMapper(atoms_mapper)
atoms.GetProperty().SetRepresentationToSurface()
atoms.GetProperty().SetInterpolationToGouraud()
atoms.GetProperty().SetAmbient(0.15)
atoms.GetProperty().SetDiffuse(0.85)
atoms.GetProperty().SetSpecular(0.1)
atoms.GetProperty().SetSpecularPower(100)
atoms.GetProperty().SetSpecularColor(1, 1, 1)
atoms.GetProperty().SetColor(1, 1, 1)

tube = vtkTubeFilter()
tube.SetInputConnection(reader.reader.GetOutputPort())
tube.SetNumberOfSides(16)
tube.SetCapping(0)
tube.SetRadius(0.2)
tube.SetVaryRadius(0)
tube.SetRadiusFactor(10)

bonds_mapper = vtkPolyDataMapper()
bonds_mapper.SetInputConnection(tube.GetOutputPort())
bonds_mapper.UseLookupTableScalarRangeOff()
bonds_mapper.SetScalarVisibility(1)
bonds_mapper.SetScalarModeToDefault()

bonds = pv.Actor()
bonds.SetMapper(bonds_mapper)
bonds.GetProperty().SetRepresentationToSurface()
bonds.GetProperty().SetInterpolationToGouraud()
bonds.GetProperty().SetAmbient(0.15)
bonds.GetProperty().SetDiffuse(0.85)
bonds.GetProperty().SetSpecular(0.1)
bonds.GetProperty().SetSpecularPower(100)
bonds.GetProperty().SetSpecularColor(1, 1, 1)
bonds.GetProperty().SetColor(1, 1, 1)
ren1.AddActor(bonds)
ren1.AddActor(atoms)
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
