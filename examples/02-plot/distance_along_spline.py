"""
.. _distance_along_spline_example:

Label based on Distance on Line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a spline and generate labels along the spline based on distance along a
spline.

This is an extension of the :ref:`create_spline_example`.

"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# %%
# Create a spline
# ~~~~~~~~~~~~~~~
# Create a spline using :func:`pyvista.Spline`.

# Make points along a spline
theta = np.linspace(-1 * np.pi, 1 * np.pi, 100)
z = np.linspace(2, -2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
points = np.column_stack((x, y, z))

# Create a spline. This automatically computes arc_length, which is the
# distance along the line.
spline = pv.Spline(points, 1000)
spline.point_data


# %%
# Determine the coordinates matching distance along a spline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we write a simple function that gets the closest point matching a distance along a
# spline and then generate labels for those points.


def get_point_along_spline(distance):
    """Return the closest point on the spline given a length along the spline."""
    idx = np.argmin(np.abs(spline.point_data['arc_length'] - distance))
    return spline.points[idx]


# distances along the spline we're interested in
dists = [0, 4, 8, 11]

# make labels
labels = []
label_points = []
for dist in dists:
    point = get_point_along_spline(dist)
    labels.append(f'Dist {dist}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})')
    label_points.append(point)

labels

# %%
# Plot with Labels
# ~~~~~~~~~~~~~~~~
# Plot the spline with labeled points

# sphinx_gallery_start_ignore
# depth field modification does not seem to work in interactive mode
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter()
pl.add_mesh(spline, scalars='arc_length', render_lines_as_tubes=True, line_width=10)
pl.add_point_labels(
    label_points,
    labels,
    always_visible=True,
    point_size=20,
    render_points_as_spheres=True,
)
pl.show_bounds()
pl.show_axes()
pl.camera_position = 'xz'
pl.show()
# %%
# .. tags:: plot
