"""
.. _gif_example:

Create a GIF Movie
~~~~~~~~~~~~~~~~~~
Generate a moving gif from an active plotter.
This example uses :meth:`~pyvista.Plotter.open_gif` and
:meth:`~pyvista.Plotter.write_frame` to create the gif.

.. note::
   Use ``lighting=False`` to reduce the size of the color space to avoid
   "jittery" GIFs, especially for the scalar bar.

"""

# sphinx_gallery_thumbnail_number = 2
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

# %%
# Create a structured grid
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Create a structured grid and make a "wave" my shifting the Z position based
# on the cartesian distance from the origin.

x = np.arange(-10, 10, 0.5)
y = np.arange(-10, 10, 0.5)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)

# Create and structured surface
grid = pv.StructuredGrid(x, y, z)
grid['Height'] = z.ravel()
grid.plot()


# %%
# Generate a GIF
# ~~~~~~~~~~~~~~
# Generate a GIF using ``off_screen=True`` parameter.

# Create a plotter object and set the scalars to the Z height
pl = pv.Plotter(notebook=False, off_screen=True)
pl.add_mesh(
    grid,
    scalars='Height',
    lighting=False,
    show_edges=True,
    clim=[-1, 1],
)

# Open a gif
pl.open_gif('wave.gif')

# Update Z and write a frame for each updated position
nframe = 15
for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
    z = np.sin(r + phase)
    # Update values inplace
    grid.points[:, -1] = z.ravel()
    grid['Height'] = z.ravel()
    # Write a frame. This triggers a render.
    pl.write_frame()

# Closes and finalizes movie
pl.close()
# %%
# .. tags:: plot
