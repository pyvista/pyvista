"""
.. _point_picking_example:

Picking points on a mesh
~~~~~~~~~~~~~~~~~~~~~~~~
This example demonstrates how to pick points on meshes using
:func:`enable_point_picking() <pyvista.Plotter.enable_point_picking>`.

"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import pyvista as pv

# sphinx_gallery_start_ignore
# picking not work in interactive plots
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Pick points on a sphere
# +++++++++++++++++++++++
#
sphere = pv.Sphere()

p = pv.Plotter()
p.add_mesh(sphere, pickable=True)
p.enable_point_picking()
p.show()

# %%
# Ignore the 3D window
# ++++++++++++++++++++
#
# In the above example, both points on the mesh and points in the 3d window can be
# selected. It is possible instead pick only points on the mesh.
sphere = pv.Sphere()

p = pv.Plotter()
p.add_mesh(sphere, pickable=True)
p.enable_point_picking(pickable_window=False)  # Make the 3D window unpickable
p.show()

# %%
# Modify which actors are pickable
# ++++++++++++++++++++++++++++++++
#
# After enabling point picking, we can modify which actors are pickable.
sphere = pv.Sphere()
cube = pv.Cube().translate([10, 10, 0])

p = pv.Plotter()
sphere_actor = p.add_mesh(sphere, pickable=True)  # initially pickable
cube_actor = p.add_mesh(cube, pickable=False)  # initially unpickable
p.enable_point_picking(pickable_window=False)

p.pickable_actors = [sphere_actor, cube_actor]  # now both are pickable
p.view_xy()
p.show()

# %%
# Pick using the left-mouse button
# ++++++++++++++++++++++++++++++++
#
sphere = pv.Sphere()

p = pv.Plotter()
p.add_mesh(sphere, pickable=True)
p.enable_point_picking(left_clicking=True)
p.show()
# %%
# .. tags:: plot
