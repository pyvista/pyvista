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

pl.add_mesh(grid.outline(), color="black")

vol = pl.add_volume(grid)

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

atoms = pl.add_mesh(
    glyph,
    color="red",
    ambient=0.15,
    diffuse=0.85,
    specular=0.1,
    style="surface",
    interpolation='Gouraud',
)

tube = dataset.tube(n_sides=16, capping=False, radius=0.2, radius_factor=10)

bonds = pl.add_mesh(
    tube,
    color="white",
    ambient=0.15,
    diffuse=0.85,
    specular=0.1,
    style="surface",
    interpolation='Gouraud',
)

pl.set_background('white')
pl.show(cpos="zx")
