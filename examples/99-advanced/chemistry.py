#!/usr/bin/env python

import pyvista as pv
from pyvista import examples

pl = pv.Plotter()

filename = examples.download_m4_total_density(load=False)
reader = pv.get_reader(filename)
reader.reader.SetHBScale(1.1)
reader.reader.SetBScale(10)
reader.reader.Update()

grid = pv.wrap(reader.reader.GetGridOutput())
dataset = pv.wrap(reader.reader.GetOutput())

bounds_actor = pv.Actor(mapper=pv.DataSetMapper(grid.outline()))
bounds_actor.prop.color = "black"
pl.add_actor(bounds_actor)

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

volume = pl.add_volume(grid)
volume.prop.SetColor(color_tf)
volume.prop.SetScalarOpacity(opacity_tf)
volume.prop.SetInterpolationTypeToLinear()

sphere = pv.SphereSource(
    center=(0, 0, 0),
    radius=1,
    theta_resolution=16,
    start_theta=0,
    end_theta=360,
    phi_resolution=16,
    start_phi=0,
    end_phi=180,
)

glyph = dataset.glyph(orient=True, scale=True, factor=0.6, geom=sphere.output, color_mode="scalar")

atoms = pv.Actor(mapper=pv.DataSetMapper(glyph))
atoms.mapper.UseLookupTableScalarRangeOff()
atoms.mapper.SetScalarVisibility(1)
atoms.mapper.SetScalarModeToDefault()
atoms.prop.SetRepresentationToSurface()
atoms.prop.SetInterpolationToGouraud()
atoms.prop.SetAmbient(0.15)
atoms.prop.SetDiffuse(0.85)
atoms.prop.SetSpecular(0.1)
atoms.prop.SetSpecularPower(100)
atoms.prop.SetSpecularColor(1, 1, 1)
atoms.prop.SetColor(1, 1, 1)
pl.add_actor(atoms)

tube = dataset.tube(n_sides=16, capping=False, radius=0.2, radius_factor=10)

bonds = pv.Actor(mapper=pv.DataSetMapper(tube))
bonds.mapper.UseLookupTableScalarRangeOff()
bonds.mapper.SetScalarVisibility(1)
bonds.mapper.SetScalarModeToDefault()
bonds.prop.SetRepresentationToSurface()
bonds.prop.SetInterpolationToGouraud()
bonds.prop.SetAmbient(0.15)
bonds.prop.SetDiffuse(0.85)
bonds.prop.SetSpecular(0.1)
bonds.prop.SetSpecularPower(100)
bonds.prop.SetSpecularColor(1, 1, 1)
bonds.prop.SetColor(1, 1, 1)
pl.add_actor(bonds)

pl.set_background('white')
pl.camera.enable_parallel_projection()
pl.camera.up = (0, 1, 0)
pl.camera.focal_point = (12, 10.5, 15)
pl.camera.position = (-70, 15, 34)
pl.show()
