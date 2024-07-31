"""
.. _create_poly:

Create PolyData
~~~~~~~~~~~~~~~

Creating a :class:`pyvista.PolyData` (surface mesh) from vertices and faces.

"""

from __future__ import annotations

import numpy as np

import pyvista as pv

# %%
# A PolyData object can be created quickly from numpy arrays.  The vertex array
# contains the locations of the points in the mesh and the face array contains
# the number of points of each face and the indices of the vertices which
# comprise that face.

# mesh points
vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

# mesh faces
faces = np.hstack(
    [
        [4, 0, 1, 2, 3],  # square
        [3, 0, 1, 4],  # triangle
        [3, 1, 2, 4],  # triangle
    ],
)

surf = pv.PolyData(vertices, faces)

# plot each face with a different color
surf.plot(
    scalars=np.arange(3),
    cpos=[-1, 1, 0.5],
    show_scalar_bar=False,
    show_edges=True,
    line_width=5,
)


# %%
# Polygonal PolyData
# ~~~~~~~~~~~~~~~~~~
# Create a three face polygonal mesh directly from points and faces.
#
# .. note::
#    It is generally more efficient to use a numpy array rather than stacking
#    lists for large meshes.

points = np.array(
    [
        [0.0480, 0.0349, 0.9982],
        [0.0305, 0.0411, 0.9987],
        [0.0207, 0.0329, 0.9992],
        [0.0218, 0.0158, 0.9996],
        [0.0377, 0.0095, 0.9992],
        [0.0485, 0.0163, 0.9987],
        [0.0572, 0.0603, 0.9965],
        [0.0390, 0.0666, 0.9970],
        [0.0289, 0.0576, 0.9979],
        [0.0582, 0.0423, 0.9974],
        [0.0661, 0.0859, 0.9941],
        [0.0476, 0.0922, 0.9946],
        [0.0372, 0.0827, 0.9959],
        [0.0674, 0.0683, 0.9954],
    ],
)


face_a = [6, 0, 1, 2, 3, 4, 5]
face_b = [6, 6, 7, 8, 1, 0, 9]
face_c = [6, 10, 11, 12, 7, 6, 13]
faces = np.concatenate((face_a, face_b, face_c))

mesh = pv.PolyData(points, faces)
mesh.plot(show_edges=True, line_width=5)

# %%
# Generate quadrilateral mesh of Sphere
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This example shows how a more complicated mesh can be defined.
# :func:`pyvista.Sphere` creates a sphere with a triangulated mesh,
# but a regular quadrilateral mesh is desired for some use case.
#
# The points are generated as a regular grid in spherical coordinates,
# plus one additional point for each pole at beginning and end of the points
# array. Here, we will used the convention that ``theta`` is the
# azimuthal angle, similar to longitude on the globe.  ``phi`` is the
# polar angle, similar to latitude on the globe.

radius = 0.5
ntheta = 9
nphi = 12
theta = np.linspace(0, 2 * np.pi, ntheta)
phi = np.linspace(0, np.pi, nphi)

# %%
# We do not want duplicate points, so remove the duplicate in theta, which
# results in 8 unique points in theta. Similarly, the poles at `phi=0` and
# `phi=pi` will be handled separately to avoid duplicate points, which
# results in 10 unique points in phi.  Remove these from the grid in spherical
# coordinates.

theta = theta[:-1]
ntheta -= 1
phi = phi[1:-1]
nphi -= 2

# %%
# Generate cartesian coordinates for points in the ``(N, 3)``
# format required by PyVista.  Note that this method results in
# the theta variable changing the fastest.

r_, phi_, theta_ = np.meshgrid([radius], phi, theta, indexing='ij')
x, y, z = pv.spherical_to_cartesian(r_, phi_, theta_)
points = np.vstack((x.ravel(), y.ravel(), z.ravel())).transpose()

# %%
# The first and last points are the poles.

points = np.insert(points, 0, [0.0, 0.0, radius])
points = np.append(points, [0.0, 0.0, -radius])

# %%
# First we will generate the cell-point connectivity similar to the
# previous examples.  At the poles, we will form triangles with the pole
# and two adjacent points from the closest ring of points at a given ``phi``
# position.  Otherwise, we will form quadrilaterals between two adjacent points
# on consecutive ``phi`` positions.
#
# The first triangle in the mesh is point id ``0``, i.e. the pole, and
# the first two points at the first ``phi`` position, id's ``1`` and ``2``.
# the next triangle contains the pole again and the next set of points,
# id's ``2`` and ``3`` and so on.  The last point in the ring, id ``8`` connects
# to the first point in the ring, ``1``, to form the last triangle.  Exclude it
# from the loop and add separately.

faces = []
for i in range(1, ntheta):
    faces.extend([3, 0, i, i + 1])

faces.extend([3, 0, ntheta, 1])

# %%
# Demonstrate the connectivity of the mesh so far.

points_to_label = tuple(range(ntheta + 1))
mesh = pv.PolyData(points, faces=faces)
pl = pv.Plotter(off_screen=True)
pl.add_mesh(mesh, show_edges=True)
pl.add_point_labels(
    mesh.points[points_to_label, :], points_to_label, font_size=30, fill_shape=False
)
pl.view_xy()
pl.show(screenshot="tmp.png")

# %%
# Next form the quadrilaterals. This process is the same except
# by connecting points across two levels of phi.  For point 1
# and point 2, these are connected to point 9 and point 10. Note
# for quadrilaterals it must be defined in a consistent direction.
# Again, the last point(s) in the theta direction connect back to the
# first point(s).

for i in range(1, ntheta):
    faces.extend([4, i, i + 1, i + ntheta + 1, i + ntheta])

faces.extend([4, ntheta, 1, ntheta + 1, ntheta * 2])

# %%
# Demonstrate the connectivity of the mesh with first quad layer.

points_to_label = tuple(range(ntheta * 2 + 1))
mesh = pv.PolyData(points, faces=faces)
pl = pv.Plotter(off_screen=True)
pl.add_mesh(mesh, show_edges=True)
pl.add_point_labels(
    mesh.points[points_to_label, :],
    points_to_label,
    font_size=30,
    fill_shape=False,
    always_visible=True,
)
pl.view_xy()
pl.show(screenshot="tmp2.png")


# %%
# Next we loop over all adjacent levels of phi to form all the quadrilaterals
# and add the layer of triangles on the ending pole.  Since we already formed
# the first layer of quadrilaterals, let's start over to make cleaner code.

faces = []
for i in range(1, ntheta):
    faces.extend([3, 0, i, i + 1])

faces.extend([3, 0, ntheta, 1])

for j in range(nphi - 1):
    for i in range(1, ntheta):
        faces.extend(
            [4, j * ntheta + i, j * ntheta + i + 1, i + (j + 1) * ntheta + 1, i + (j + 1) * ntheta]
        )

    faces.extend([4, (j + 1) * ntheta, j * ntheta + 1, (j + 1) * ntheta + 1, (j + 2) * ntheta])

for i in range(1, ntheta):
    faces.extend([3, nphi * ntheta + 1, (nphi - 1) * ntheta + i, (nphi - 1) * ntheta + i + 1])

faces.extend([3, nphi * ntheta + 1, nphi * ntheta, (nphi - 1) * ntheta + 1])

# %%
# All the point labels are messy when plotted, so don't add to the final plot.

mesh = pv.PolyData(points, faces=faces)
pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True)
pl.show(screenshot="tmp3.png")

# %%
# .. tags:: load
