"""
.. _follower_actor_example:

Follower Actor
~~~~~~~~~~~~~~

This example demonstrates how to use the :class:`pyvista.Follower` actor,
which is an actor that always faces the camera. This is useful for creating
billboarding effects and screen-aligned labels.

"""

from __future__ import annotations

import pyvista as pv

################################################################################
# Create a simple scene with a follower actor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Create a follower actor that will always face the camera even as you
# rotate the scene.

# Create a simple text mesh
text = pv.Text3D('Always Facing You', depth=0.1)

# Create a follower actor with this text
plotter = pv.Plotter()
mapper = pv.DataSetMapper(text)
follower = pv.Follower(mapper=mapper)
follower.prop.color = 'blue'

# IMPORTANT: Set the camera for the follower
follower.camera = plotter.camera

# Add the follower to the scene
plotter.add_actor(follower)

# Add some reference geometry that won't rotate
plotter.add_mesh(pv.Cube(center=(2, 0, 0)), color='red', label='Static Cube')

# Show the scene
plotter.show_legend()
plotter.show()

################################################################################
# Multiple Followers in a Scene
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# You can create multiple follower actors in a scene. This is useful
# for creating labels that always face the camera.

# Create a plotter
plotter = pv.Plotter()

# Create some 3D points
points = [
    (0, 0, 0),
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
]

labels = ['Origin', 'X-axis', 'Y-axis', 'Z-axis']
colors = ['black', 'red', 'green', 'blue']

# Create follower actors for each label
for point, label, color in zip(points, labels, colors):
    # Create text mesh
    text = pv.Text3D(label, depth=0.05)

    # Create mapper and follower
    mapper = pv.DataSetMapper(text)
    follower = pv.Follower(mapper=mapper)

    # Set properties
    follower.prop.color = color
    follower.position = point
    follower.scale = 0.5
    follower.camera = plotter.camera

    # Add to plotter
    plotter.add_actor(follower)

    # Also add a small sphere at each point
    plotter.add_mesh(pv.Sphere(radius=0.1, center=point), color=color)

# Add coordinate axes for reference
plotter.add_axes()

# Show the scene
plotter.show()

################################################################################
# Follower with Different Geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Followers can use any geometry, not just text. Here's an example
# with different shapes that always face the camera.

plotter = pv.Plotter()

# Create different geometries
geometries = [
    (pv.Arrow(), 'Arrow', (0, 0, 0)),
    (pv.Cone(), 'Cone', (3, 0, 0)),
    (pv.Cylinder(), 'Cylinder', (0, 3, 0)),
    (pv.Box(), 'Box', (3, 3, 0)),
]

for geom, name, pos in geometries:
    # Create follower
    mapper = pv.DataSetMapper(geom)
    follower = pv.Follower(mapper=mapper)
    follower.position = pos
    follower.scale = 0.5
    follower.camera = plotter.camera

    # Add to scene
    plotter.add_actor(follower)

    # Add label
    plotter.add_point_labels([pos], [name], point_size=10)

# Add a grid for reference
plotter.show_grid()
plotter.show()
