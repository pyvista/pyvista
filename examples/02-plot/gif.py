"""
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
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars=z.ravel(), smooth_shading=True)

print('Orient the view, then press "q" to close window and produce movie')

# setup camera and close
plotter.show(auto_close=False)

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
    plotter.write_frame()  # this will trigger the render

    # otherwise, when not writing frames, render with:
    # plotter.render()

# Close movie and delete object
plotter.close()
