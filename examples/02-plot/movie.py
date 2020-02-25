"""
Create a MP4 Movie
~~~~~~~~~~~~~~~~~~

Create an animated MP4 movie of a rendering scene
"""


import pyvista as pv
import numpy as np

filename = "sphere-shrinking.mp4"

mesh = pv.Sphere()
mesh.cell_arrays["data"] = np.random.random(mesh.n_cells)

plotter = pv.Plotter()
# Open a movie file
plotter.open_movie(filename)

# Add initial mesh
plotter.add_mesh(mesh, scalars="data", clim=[0, 1])
# Add outline for shrinking reference
plotter.add_mesh(mesh.outline_corners())

print('Orient the view, then press "q" to close window and produce movie')

# Render and do NOT close
plotter.show(auto_close=False)

# Run through each frame
plotter.write_frame()  # write initial data

# Update scalars on each frame
for i in range(100):
    random_points = np.random.random(mesh.points.shape)
    mesh.points = random_points * 0.01 + mesh.points * 0.99
    mesh.points -= mesh.points.mean(0)
    mesh.cell_arrays["data"] = np.random.random(mesh.n_cells)
    plotter.write_frame()  # Write this frame

# Be sure to close the plotter when finished
plotter.close()

###############################################################################
# .. figure:: ../../images/auto-generated/sphere-shrinking.mp4
