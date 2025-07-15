"""
.. _collision_example:

Collision
~~~~~~~~~
Perform a collision detection between two meshes.

This example use the :meth:`~pyvista.PolyDataFilters.collision`
filter to detect the faces from one sphere colliding with another
sphere.

.. note::
   Due to the nature of the :vtk:`vtkCollisionDetectionFilter`,
   repeated uses of this method will be slower that using the
   :vtk:`vtkCollisionDetectionFilter` directly.  The first
   update of the filter creates two instances of :vtk:`vtkOBBTree`,
   which can be subsequently updated by modifying the transform or
   matrix of the input meshes.

   This method assumes no transform and is easier to use for
   single collision tests, but it is recommended to use a
   combination of ``pyvista`` and ``vtk`` for rapidly computing
   repeated collisions.  See the `Collision Detection Example
   <https://kitware.github.io/vtk-examples/site/Python/Visualization/CollisionDetection/>`_


"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "pyvista",
# ]
# ///

from __future__ import annotations

import numpy as np

import pyvista as pv

pv.set_plot_theme('document')


# %%
# Create the main mesh and the secondary "moving" mesh
#
# Collision faces will be plotted on this sphere, and to do so we
# initialize an initial ``"collisions"`` mask.
sphere0 = pv.Sphere()
sphere0['collisions'] = np.zeros(sphere0.n_cells, dtype=bool)

# This mesh will be the moving mesh
sphere1 = pv.Sphere(radius=0.6, center=(-1, 0, 0))

# %%
# Set up the plotter open a movie, and write a frame after moving the sphere.
#

pl = pv.Plotter()
pl.enable_hidden_line_removal()
pl.add_mesh(sphere0, scalars='collisions', show_scalar_bar=False, cmap='bwr')
pl.camera_position = 'xz'
pl.add_mesh(sphere1, style='wireframe', color='green', line_width=5)

# for this example
pl.open_gif('collision_movie.gif')

# alternatively, to disable movie generation:
# pl.show(auto_close=False, interactive=False)

delta_x = 0.05
for _ in range(int(2 / delta_x)):
    sphere1.translate([delta_x, 0, 0], inplace=True)
    col, n_contacts = sphere0.collision(sphere1)

    collision_mask = np.zeros(sphere0.n_cells, dtype=bool)
    if n_contacts:
        collision_mask[col['ContactCells']] = True
    sphere0['collisions'] = collision_mask
    pl.write_frame()

    # alternatively, disable movie plotting and simply render the image
    # pl.render()

pl.close()
# %%
# .. tags:: filter
