"""
.. _create_kochanek_spline_example:

Create a Kochanek Spline
~~~~~~~~~~~~~~~~~~~~~~~~

Create a Kochanek spline/polyline from a numpy array of XYZ vertices.
"""

import pyvista as pv
import numpy as np


###############################################################################
# Create a dataset to plot


def make_points():
    """Helper to make XYZ points"""
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 6)
    z = np.linspace(-2, 2, 6)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.column_stack((x, y, z))


points = make_points()
points[0:5, :]

###############################################################################
# Interpolate those points onto a parametric Kochanek spline

# Create Kochanek spline with 6 interpolation points
p = pv.Plotter(shape=(3, 5))

c = [-1.0, -0.5, 0.0, 0.5, 1.0]
for i in range(5):
    kochanek_spline = pv.KochanekSpline(
        points, continuity=[c[i], c[i], c[i]], n_points=1000
    )
    p.subplot(0, i)
    p.add_text("c = " + str(c[i]))
    p.add_mesh(kochanek_spline, color="k", point_size=10)
    p.add_mesh(
        pv.PolyData(points),
        color="k",
        point_size=10,
        render_points_as_spheres=True,
    )

t = [-1.0, -0.5, 0.0, 0.5, 1.0]
for i in range(5):
    kochanek_spline = pv.KochanekSpline(
        points, tension=[t[i], t[i], t[i]], n_points=1000
    )
    p.subplot(1, i)
    p.add_text("t = " + str(t[i]))
    p.add_mesh(kochanek_spline, color="k")
    p.add_mesh(
        pv.PolyData(points),
        color="k",
        point_size=10,
        render_points_as_spheres=True,
    )

b = [-1.0, -0.5, 0.0, 0.5, 1.0]
for i in range(5):
    kochanek_spline = pv.KochanekSpline(points, bias=[b[i], b[i], b[i]], n_points=1000)
    p.subplot(2, i)
    p.add_text("b = " + str(b[i]))
    p.add_mesh(kochanek_spline, color="k")
    p.add_mesh(
        pv.PolyData(points),
        color="k",
        point_size=10,
        render_points_as_spheres=True,
    )

p.show(cpos="xy")
