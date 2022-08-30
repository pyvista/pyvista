"""
.. _mesh_picking_example:

Picking Meshes
~~~~~~~~~~~~~~
This example demonstrates how to pick meshes using
:func:`enable_mesh_picking() <pyvista.Plotter.enable_mesh_picking>`.

"""

import pyvista as pv

###############################################################################
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


###############################################################################
# Pick using the left-mouse button
# ++++++++++++++++++++++++++++++++
# Pick using the left mouse button and trigger a callback that "shrinks" the
# mesh each time it's selected.


def callback(mesh):
    """Shrink the mesh each time it's clicked."""
    shrunk = mesh.shrink(0.9)
    mesh.copy_from(shrunk)  # must operate "in-place" by overwrite


pl = pv.Plotter()
pl.add_mesh(sphere, color='r')
pl.add_mesh(cube, color='b')
pl.enable_mesh_picking(callback=callback, left_clicking=True, show=False)
pl.show()
