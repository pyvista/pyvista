"""
.. _gif_movie_example:

Create a GIF Movie
~~~~~~~~~~~~~~~~~~

Generate a moving gif from an active plotter
"""

import pyvista as pv
import numpy as np

x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x ** 2 + y ** 2)
z = np.sin(r)

# Create and structured surface
grid = pv.StructuredGrid(x, y, z)

# Create a plotter object and set the scalars to the Z height
plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.add_mesh(grid, scalars=z.ravel(), smooth_shading=True)

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

    # must update normals when smooth shading is enabled
    plotter.mesh.compute_normals(cell_normals=False, inplace=True)
    plotter.render()
    plotter.write_frame()

# Closes and finalizes movie
plotter.close()
