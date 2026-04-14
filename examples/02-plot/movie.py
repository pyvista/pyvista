"""
.. _movie_example:

Create a MP4 Movie
~~~~~~~~~~~~~~~~~~

Create an animated MP4 movie of a rendering scene.
This example uses :meth:`~pyvista.Plotter.open_movie` and
:meth:`~pyvista.Plotter.write_frame` to create the movie.

.. Note::
    This movie will appear static since MP4 movies will not be
    rendered on a sphinx gallery example.

"""

from __future__ import annotations

import numpy as np
import pyvista as pv

filename = 'sphere-shrinking.mp4'

# Create a sphere with random data. Seed the rng to make it reproducible.
rng = np.random.default_rng(seed=0)
mesh = pv.Sphere()
mesh.cell_data['data'] = rng.random(mesh.n_cells)

pl = pv.Plotter()
# Open a movie file
pl.open_movie(filename)

# Add initial mesh
pl.add_mesh(mesh, scalars='data', clim=[0, 1])
# Add outline for shrinking reference
pl.add_mesh(mesh.outline_corners())

pl.show(auto_close=False)  # only necessary for an off-screen movie

# Run through each frame
pl.write_frame()  # write initial data

# Update scalars on each frame
for i in range(100):
    random_points = rng.random(mesh.points.shape)
    mesh.points = random_points * 0.01 + mesh.points * 0.99
    mesh.points -= mesh.points.mean(0)
    mesh.cell_data['data'] = rng.random(mesh.n_cells)
    pl.add_text(f'Iteration: {i}', name='time-label')
    pl.write_frame()  # Write this frame

# Be sure to close the plotter when finished
pl.close()
# %%
# .. tags:: plot
