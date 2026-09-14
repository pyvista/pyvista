"""
.. _curvatures_adjust_edges:

curvatures Adjust Edges
~~~~~~~~~~~~~~~~~~~~~~~

This example is ported from `CurvaturesAdjustEdges <https://examples.vtk.org/site/Python/PolyData/CurvaturesAdjustEdges/>`_ .
"""

from __future__ import annotations

import numpy as np

import pyvista as pv

source = (
    pv.ParametricRandomHills(
        random_seed=1, number_of_hills=30, u_res=51, v_res=51, texture_coordinates=True
    )
    .translate((0.0, 5.0, 15.0))
    .rotate_x(-90.0)
)

source['Gauss_Curvature'] = source.adjusted_edge_curvature("gaussian")
source['Mean_Curvature'] = source.adjusted_edge_curvature("mean")

# Absolute curvature values less than 1.0e-08 will be set to zero.

epsilon = 1.0e-08
source['Gauss_Curvature'] = np.where(
    abs(source['Gauss_Curvature']) < epsilon, 0, source['Gauss_Curvature']
)
source['Mean_Curvature'] = np.where(
    abs(source['Mean_Curvature']) < epsilon, 0, source['Mean_Curvature']
)

# Let's visualize what we have done.

plotter = pv.Plotter(shape=(1, 2), window_size=(1024, 512))

lookup_table = pv.LookupTable('coolwarm', n_values=256)

curvature_name = 'Gauss_Curvature'
plotter.subplot(0, 0)
curvature_title = curvature_name.replace('_', '\n')

source.set_active_scalars(curvature_name)

mapper = pv.DataSetMapper(source)
mapper.scalar_map_mode = 'point_field'
mapper.SelectColorArray(curvature_name)
mapper.scalar_range = (np.min(source.active_scalars), np.max(source.active_scalars))
mapper.lookup_table = lookup_table

actor = pv.Actor(mapper=mapper)

plotter.add_actor(actor)
plotter.set_background([82, 87, 110])
text_actor = plotter.add_text(curvature_title, position=(250, 16))
text_actor.prop.font_size = 24
text_actor.prop.justification_horizontal = "center"
text_actor.prop.color = "white"
plotter.add_scalar_bar(
    title=curvature_title,
    unconstrained_font_size=True,
    mapper=mapper,
    n_labels=5,
    position_x=0.85,
    position_y=0.1,
    vertical=True,
    color='white',
)
renderer = plotter.renderers[0]

camera = renderer.camera
camera.elevation = 60
renderer.reset_camera()

curvature_name = 'Mean_Curvature'
plotter.subplot(0, 1)
curvature_title = curvature_name.replace('_', '\n')

source.set_active_scalars(curvature_name)

mapper = pv.DataSetMapper(source)
mapper.scalar_map_mode = 'point_field'
mapper.SelectColorArray(curvature_name)
mapper.scalar_range = (np.min(source.active_scalars), np.max(source.active_scalars))
mapper.lookup_table = lookup_table

actor = pv.Actor(mapper=mapper)

plotter.add_actor(actor)
plotter.set_background([82, 87, 110])
text_actor = plotter.add_text(curvature_title, position=(250, 16))
text_actor.prop.font_size = 24
text_actor.prop.justification_horizontal = "center"
text_actor.prop.color = "white"
plotter.add_scalar_bar(
    title=curvature_title,
    unconstrained_font_size=True,
    mapper=mapper,
    n_labels=5,
    position_x=0.85,
    position_y=0.1,
    vertical=True,
    color='white',
)
renderer = plotter.renderers[1]


renderer.camera = camera
renderer.reset_camera()

plotter.add_camera_orientation_widget()
plotter.show()
