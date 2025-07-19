"""
.. _mesh_picking_example:

Picking Meshes
~~~~~~~~~~~~~~
This example demonstrates how to pick meshes using
:func:`enable_mesh_picking() <pyvista.Plotter.enable_mesh_picking>`.

"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

import pyvista as pv

# %%
# Pick either a cube or a sphere using "p"
# ++++++++++++++++++++++++++++++++++++++++
#

sphere = pv.Sphere(center=(1, 0, 0))
cube = pv.Cube()

pl = pv.Plotter()
pl.add_mesh(sphere, color='r')
pl.add_mesh(cube, color='b')
pl.enable_mesh_picking()
pl.show()


# %%
# Deform the mesh after picking
# +++++++++++++++++++++++++++++
# Pick to trigger a callback that "shrinks" the mesh each time it's selected.


def callback(mesh):
    """Shrink the mesh each time it's clicked."""
    shrunk = mesh.shrink(0.9)
    mesh.copy_from(shrunk)  # make operation "in-place" by replacing the original mesh


pl = pv.Plotter()
pl.add_mesh(sphere, color='r')
pl.add_mesh(cube, color='b')
pl.enable_mesh_picking(callback=callback, show=False)
pl.show()


# %%
# Pick based on Actors
# ++++++++++++++++++++
# Return the picked actor to the callback

pl = pv.Plotter()
pl.add_mesh(pv.Cone(center=(0, 0, 0)), name='Cone')
pl.add_mesh(pv.Cube(center=(1, 0, 0)), name='Cube')
pl.add_mesh(pv.Sphere(center=(1, 1, 0)), name='Sphere')
pl.add_mesh(pv.Cylinder(center=(0, 1, 0)), name='Cylinder')


def reset():
    for a in pl.renderer.actors.values():
        if isinstance(a, pv.Actor):
            a.prop.color = 'lightblue'
            a.prop.show_edges = False


def callback(actor):
    reset()
    actor.prop.color = 'green'
    actor.prop.show_edges = True


pl.enable_mesh_picking(callback, use_actor=True, show=False)
pl.show()
# %%
# .. tags:: plot
