"""
.. _create_kochanek_spline_example:

Create a Kochanek Spline
~~~~~~~~~~~~~~~~~~~~~~~~

Create a Kochanek spline/polyline from a numpy array of XYZ vertices.
Uses :func:`pyvista.KochanekSpline`.
"""

# sphinx_gallery_start_ignore
from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import numpy as np

import pyvista as pv

# %%
# Create a dataset to plot


def make_points():
    """Make XYZ points."""
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 6)
    z = np.linspace(-2, 2, 6)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.column_stack((x, y, z))


points = make_points()
points[0:5, :]

# %%
# Interpolate those points onto a parametric Kochanek spline

# Create Kochanek spline with 6 interpolation points
pl = pv.Plotter(shape=(3, 5))

c = [-1.0, -0.5, 0.0, 0.5, 1.0]
for i in range(5):
    kochanek_spline = pv.KochanekSpline(points, continuity=[c[i], c[i], c[i]], n_points=1000)
    pl.subplot(0, i)
    pl.add_text('c = ' + str(c[i]))
    pl.add_mesh(kochanek_spline, color='k', point_size=10)
    pl.add_mesh(
        pv.PolyData(points),
        color='k',
        point_size=10,
        render_points_as_spheres=True,
    )

t = [-1.0, -0.5, 0.0, 0.5, 1.0]
for i in range(5):
    kochanek_spline = pv.KochanekSpline(points, tension=[t[i], t[i], t[i]], n_points=1000)
    pl.subplot(1, i)
    pl.add_text('t = ' + str(t[i]))
    pl.add_mesh(kochanek_spline, color='k')
    pl.add_mesh(
        pv.PolyData(points),
        color='k',
        point_size=10,
        render_points_as_spheres=True,
    )

b = [-1.0, -0.5, 0.0, 0.5, 1.0]
for i in range(5):
    kochanek_spline = pv.KochanekSpline(points, bias=[b[i], b[i], b[i]], n_points=1000)
    pl.subplot(2, i)
    pl.add_text('b = ' + str(b[i]))
    pl.add_mesh(kochanek_spline, color='k')
    pl.add_mesh(
        pv.PolyData(points),
        color='k',
        point_size=10,
        render_points_as_spheres=True,
    )

pl.show(cpos='xy')
# %%
# .. tags:: load
