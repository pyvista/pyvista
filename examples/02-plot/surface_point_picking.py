"""
.. _surface_point_picking_example:

Picking a Point on the Surface of a Mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example demonstrates how to pick meshes using
:func:`enable_surface_point_picking() <pyvista.Plotter.enable_surface_point_picking>`.

This allows you to pick points on the surface of a mesh.

"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

import pyvista as pv

# sphinx_gallery_start_ignore
# picking does not work on interactive examples
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Create a mesh and enable picking using the default settings.

cube = pv.Cube()

pl = pv.Plotter()
pl.add_mesh(cube, show_edges=True)
pl.enable_surface_point_picking()
pl.show()


# %%
# Enable a callback that creates a cube at the right-clicked point and add a
# label at the point as well it.


def callback(point):
    """Create a cube and a label at the click point."""
    mesh = pv.Cube(center=point, x_length=0.05, y_length=0.05, z_length=0.05)
    pl.add_mesh(mesh, style='wireframe', color='r')
    pl.add_point_labels(point, [f'{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}'])


pl = pv.Plotter()
pl.add_mesh(cube, show_edges=True)
pl.enable_surface_point_picking(callback=callback, show_point=False)
pl.show()
# %%
# .. tags:: plot
