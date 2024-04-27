#!/usr/bin/env python
from vtkmodules.vtkImagingCore import vtkImageShiftScale

import pyvista as pv
from pyvista import examples

pl = pv.Plotter()

camera = pv.Camera()
camera.ParallelProjectionOn()
camera.SetViewUp(0, 1, 0)
camera.SetFocalPoint(12, 10.5, 15)
camera.SetPosition(-70, 15, 34)
pl.renderer.SetActiveCamera(camera)
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

reader_shift_scale = vtkImageShiftScale()
reader_shift_scale.SetInputData(reader.reader.GetGridOutput())
reader_shift_scale.SetShift(min_ * -1)
reader_shift_scale.SetScale(255 / (max_ - min_))
reader_shift_scale.SetOutputScalarTypeToUnsignedChar()

bounds = pv.outline_algorithm(reader.reader.GetGridOutput())

bounds_mapper = pv.DataSetMapper(bounds.GetOutputPort())

bounds_actor = pv.Actor()
bounds_actor.SetMapper(bounds_mapper)
bounds_actor.GetProperty().SetColor(0, 0, 0)

dataset = pv.wrap(reader.reader.GetGridOutput())

contour_mapper = pv.DataSetMapper(dataset.contour(isosurfaces=[0, 0.05]))
contour_mapper.SetScalarRange(0, 0.1)
contour_mapper.GetLookupTable().SetHueRange(0.32, 0)

contourActor = pv.Actor()
contourActor.SetMapper(contour_mapper)
contourActor.GetProperty().SetOpacity(0.5)

# Create transfer mapping scalar value to opacity
lut = pv.LookupTable()
opacity_tf = lut.to_opacity_tf()
opacity_tf.RemoveAllPoints()
opacity_tf.AddPoint(0, 0.01)
opacity_tf.AddPoint(255, 0.35)
opacity_tf.ClampingOn()

# Create transfer mapping scalar value to color
color_tf = lut.to_color_tf()
color_tf.RemoveAllPoints()
color_tf.AddHSVPoint(0.0, 0.66, 1.0, 1.0)
color_tf.AddHSVPoint(50.0, 0.33, 1.0, 1.0)
color_tf.AddHSVPoint(100.0, 0.00, 1.0, 1.0)

# The property describes how the data will look
volume_property = pv.VolumeProperty()
volume_property.SetColor(color_tf)
volume_property.SetScalarOpacity(opacity_tf)
volume_property.SetInterpolationTypeToLinear()

# The mapper knows how to render the data
volume_mapper = pv.FixedPointVolumeRayCastMapper()
volume_mapper.SetInputConnection(reader_shift_scale.GetOutputPort())

# The volume holds the mapper and the property and
# can be used to position/orient the volume
volume = pv.Volume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

pl.renderer.AddVolume(volume)

# pl.renderer AddActor contourActor
pl.renderer.AddActor(bounds_actor)

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

dataset = pv.wrap(reader.reader.GetOutput())
glyph = dataset.glyph(orient=True, scale=True, factor=0.6, geom=sphere.output, color_mode="scalar")

atoms_mapper = pv.DataSetMapper(glyph)
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

dataset = pv.wrap(reader.reader.GetOutput())
tube = dataset.tube(n_sides=16, capping=False, radius=0.2, radius_factor=10)

bonds_mapper = pv.DataSetMapper(tube)
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
pl.renderer.AddActor(bonds)
pl.renderer.AddActor(atoms)
####################################################
pl.renderer.SetBackground(1, 1, 1)
pl.renderer.ResetCamera()

pl.show()
