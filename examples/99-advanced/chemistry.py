#!/usr/bin/env python

import pyvista as pv
from pyvista import examples

pl = pv.Plotter()

pl.camera.enable_parallel_projection()
pl.camera.up = (0, 1, 0)
pl.camera.focal_point = (12, 10.5, 15)
pl.camera.position = (-70, 15, 34)

filename = examples.download_m4_total_density(load=False)
reader = pv.get_reader(filename)
reader.reader.SetHBScale(1.1)
reader.reader.SetBScale(10)
reader.reader.Update()

bounds = pv.outline_algorithm(reader.reader.GetGridOutput())

bounds_mapper = pv.DataSetMapper(bounds.GetOutputPort())

bounds_actor = pv.Actor(mapper=bounds_mapper)
bounds_actor.prop.color = "black"

grid = pv.wrap(reader.reader.GetGridOutput())

contour_mapper = pv.DataSetMapper(grid.contour(isosurfaces=[0, 0.05]))
contour_mapper.SetScalarRange(0, 0.1)
contour_mapper.GetLookupTable().SetHueRange(0.32, 0)

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


volume = pl.add_volume(grid)
volume.SetProperty(volume_property)

pl.add_actor(bounds_actor)

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

atoms = pv.Actor(mapper=atoms_mapper)
atoms.prop.SetRepresentationToSurface()
atoms.prop.SetInterpolationToGouraud()
atoms.prop.SetAmbient(0.15)
atoms.prop.SetDiffuse(0.85)
atoms.prop.SetSpecular(0.1)
atoms.prop.SetSpecularPower(100)
atoms.prop.SetSpecularColor(1, 1, 1)
atoms.prop.SetColor(1, 1, 1)

tube = dataset.tube(n_sides=16, capping=False, radius=0.2, radius_factor=10)

bonds_mapper = pv.DataSetMapper(tube)
bonds_mapper.UseLookupTableScalarRangeOff()
bonds_mapper.SetScalarVisibility(1)
bonds_mapper.SetScalarModeToDefault()

bonds = pv.Actor(mapper=bonds_mapper)
bonds.prop.SetRepresentationToSurface()
bonds.prop.SetInterpolationToGouraud()
bonds.prop.SetAmbient(0.15)
bonds.prop.SetDiffuse(0.85)
bonds.prop.SetSpecular(0.1)
bonds.prop.SetSpecularPower(100)
bonds.prop.SetSpecularColor(1, 1, 1)
bonds.prop.SetColor(1, 1, 1)

pl.add_actor(bonds)
pl.add_actor(atoms)
pl.set_background('white')
pl.renderer.ResetCamera()

pl.show()
