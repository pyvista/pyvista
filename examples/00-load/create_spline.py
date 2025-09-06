"""
.. _create_spline_example:

Creating a Spline
~~~~~~~~~~~~~~~~~

Create a spline/polyline from a numpy array of XYZ vertices using
:func:`pyvista.Spline`.
"""

from __future__ import annotations

import numpy as np

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv

# %%
# Create a dataset to plot


def make_points():
    """Make XYZ points."""
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.column_stack((x, y, z))


points = make_points()
points[0:5, :]

# %%
# Now let's make a function that can create line cells on a
# :class:`pyvista.PolyData` mesh given that the points are in order for the
# segments they make.


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


line = lines_from_points(points)
line

# %%
line['scalars'] = np.arange(line.n_points, dtype=float)
tube = line.tube(radius=0.1)
tube.plot(smooth_shading=True)


# %%
# That tube has sharp edges at each line segment. This can be mitigated by
# creating a single PolyLine cell for all of the points


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


polyline = polyline_from_points(points)
polyline['scalars'] = np.arange(polyline.n_points, dtype=float)
tube = polyline.tube(radius=0.1)
tube.plot(smooth_shading=True)


# %%
# You could also interpolate those points onto a parametric spline

# Create spline with 1000 interpolation points
spline = pv.Spline(points, 1000)

# %%
# Plot spline as a tube

# add scalars to spline and plot it
spline['scalars'] = np.arange(spline.n_points, dtype=float)
tube = spline.tube(radius=0.1)
tube.plot(smooth_shading=True)

# %%
# The spline can also be plotted as a plain line

# generate same spline with 400 interpolation points
spline = pv.Spline(points, 400)

# plot without scalars
spline.plot(line_width=4, color='k')


# %%
# The radius of the tube can be modulated with scalars

spline['theta'] = 0.4 * np.arange(len(spline.points))
spline['radius'] = np.abs(np.sin(spline['theta']))
tube = spline.tube(scalars='radius', absolute=True)
tube.plot(scalars='theta', smooth_shading=True)


# %%
# Ribbons
# +++++++
#
# Ayy of the lines from the examples above can be used to create ribbons.
# Take a look at the :func:`pyvista.PolyDataFilters.ribbon` filter.

ribbon = spline.compute_arc_length().ribbon(width=0.75, scalars='arc_length')
ribbon.plot(color=True)
# %%
# .. tags:: load
