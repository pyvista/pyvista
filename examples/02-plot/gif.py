"""
.. _gif_movie_example:

Create a GIF Movie
~~~~~~~~~~~~~~~~~~
Generate a moving gif from an active plotter.

.. note::
   Use ``lighting=False`` to reduce the size of the color space to avoid
   "jittery" GIFs, especially for the scalar bar.

"""
# sphinx_gallery_thumbnail_number = 2

import numpy as np

import pyvista as pv

###############################################################################
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
grid.plot()


###############################################################################
# Generate a GIF
# ~~~~~~~~~~~~~~
# Generate a GIF using ``off_screen=True`` parameter.

# Create a plotter object and set the scalars to the Z height
plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.add_mesh(
    grid,
    scalars=z.ravel(),
    lighting=False,
    show_edges=True,
    scalar_bar_args={"title": "Height"},
    clim=[-1, 1],
)

# Open a gif
plotter.open_gif("wave.gif")

pts = grid.points.copy()

# Update Z and write a frame for each updated position
nframe = 15
for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
    z = np.sin(r + phase)
    pts[:, -1] = z.ravel()
    plotter.update_coordinates(pts, render=False)
    plotter.update_scalars(z.ravel(), render=False)

    # Write a frame. This triggers a render.
    plotter.write_frame()

# Closes and finalizes movie
plotter.close()
