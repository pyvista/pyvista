"""
.. _ref_create_spline:

Creating a Spline
~~~~~~~~~~~~~~~~~

Create a spline from numpy arrays
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import numpy as np


################################################################################
# Create a dataset to plot

# data
n_points = 100
theta = np.linspace(-4 * np.pi, 4 * np.pi, n_points)
z = np.linspace(-2, 2, n_points)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
points = np.column_stack((x, y, z))

################################################################################
# Now pass the NumPy data to PyVista

# Create spline with 1000 interpolation points
spline = pv.Spline(points, 1000)

################################################################################
# Plot spline as a tube

# add scalars to spline and plot it
spline['scalars'] = np.arange(spline.n_points)
tube = spline.tube(radius=0.1)
tube.plot(smooth_shading=True)

################################################################################
# Line can also be plotted as a plain line

# generate same spline with 400 interpolation points
spline = pv.Spline(points, 400)

# plot without scalars
spline.plot(line_width=4, color='k')
