"""
.. _slice_example:

Slicing
~~~~~~~

Extract thin planar slices from a volume.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

# %%
# PyVista meshes have several slicing filters bound directly to all datasets.
# These filters allow you to slice through a volumetric dataset to extract and
# view sections through the volume of data.
#
# One of the most common slicing filters used in PyVista is the
# :func:`pyvista.DataObjectFilters.slice_orthogonal` filter which creates three
# orthogonal slices through the dataset parallel to the three Cartesian planes.
# For example, let's slice through the sample geostatistical training image
# volume. First, load up the volume and preview it:

mesh = examples.load_channels()
# define a categorical colormap
cmap = plt.get_cmap('viridis', 4)

mesh.plot(cmap=cmap)

# %%
# Note that this dataset is a 3D volume and there might be regions within this
# volume that we would like to inspect. We can create slices through the mesh
# to gain further insight about the internals of the volume.

slices = mesh.slice_orthogonal()

slices.plot(cmap=cmap)


# %%
# The orthogonal slices can be easily translated throughout the volume:

slices = mesh.slice_orthogonal(x=20, y=20, z=30)
slices.plot(cmap=cmap)
# %%
# We can also add just a single slice of the volume by specifying the origin
# and normal of the slicing plane with the :func:`pyvista.DataObjectFilters.slice`
# filter:

# Single slice - origin defaults to the center of the mesh
single_slice = mesh.slice(normal=[1, 1, 0])

pl = pv.Plotter()
pl.add_mesh(mesh.outline(), color='k')
pl.add_mesh(single_slice, cmap=cmap)
pl.show()
# %%
# Adding slicing planes uniformly across an axial direction can also be
# automated with the :func:`pyvista.DataObjectFilters.slice_along_axis` filter:

slices = mesh.slice_along_axis(n=7, axis='y')

slices.plot(cmap=cmap)


# %%
# Slice Along Line
# ++++++++++++++++
#
# We can also slice a dataset along a :func:`pyvista.Spline` or :func:`pyvista.Line`
# using the :func:`pyvista.DataObjectFilters.slice_along_line` filter.
#
# First, define a line source through the dataset of interest. Please note
# that this type of slicing is computationally expensive and might take a while
# if there are a lot of points in the line - try to keep the resolution of
# the line low.

model = examples.load_channels()


def path(y):
    """Equation: x = a(y-h)^2 + k"""
    a = 110.0 / 160.0**2
    x = a * y**2 + 0.0
    return x, y


x, y = path(np.arange(model.bounds.y_min, model.bounds.y_max, 15.0))
zo = np.linspace(9.0, 11.0, num=len(y))
points = np.c_[x, y, zo]
spline = pv.Spline(points, n_points=15)
spline


# %%
# Then run the filter
slc = model.slice_along_line(spline)
slc

# %%

pl = pv.Plotter()
pl.add_mesh(slc, cmap=cmap)
pl.add_mesh(model.outline())
pl.show(cpos=[1, -1, 1])


# %%
# Multiple Slices in Vector Direction
# +++++++++++++++++++++++++++++++++++
#
# Slice a mesh along a vector direction perpendicularly.

mesh = examples.download_brain()

# Create vector
vec = np.array([1.0, 2.0, 1.0])
# Normalize the vector
normal = vec / np.linalg.norm(vec)

# Make points along that vector for the extent of your slices
a = mesh.center + normal * mesh.length / 3.0
b = mesh.center - normal * mesh.length / 3.0

# Define the line/points for the slices
n_slices = 5
line = pv.Line(a, b, resolution=n_slices)

# Generate all of the slices
slices = pv.MultiBlock()
for point in line.points:
    slices.append(mesh.slice(normal=normal, origin=point))

# %%

pl = pv.Plotter()
pl.add_mesh(mesh.outline(), color='k')
pl.add_mesh(slices, opacity=0.75)
pl.add_mesh(line, color='red', line_width=5)
pl.show()


# %%
# Slice At Different Bearings
# +++++++++++++++++++++++++++
#
# From `pyvista-support#23 <https://github.com/pyvista/pyvista-support/issues/23>`_
#
# An example of how to get many slices at different bearings all centered
# around a user-chosen location.
#
# Create a point to orient slices around
ranges = np.ptp(np.array(model.bounds).reshape(-1, 2), axis=1)
point = np.array(model.center) - ranges * 0.25

# %%
# Now generate a few normal vectors to rotate a slice around the z-axis.
# Use equation for circle since its about the Z-axis.
increment = np.pi / 6.0
# use a container to hold all the slices
slices = pv.MultiBlock()  # treat like a dictionary/list
for theta in np.arange(0, np.pi, increment):
    normal = np.array([np.cos(theta), np.sin(theta), 0.0]).dot(np.pi / 2.0)
    name = f'Bearing: {np.rad2deg(theta):.2f}'
    slices[name] = model.slice(origin=point, normal=normal)
slices

# %%
# And now display it.
pl = pv.Plotter()
pl.add_mesh(slices, cmap=cmap)
pl.add_mesh(model.outline())
pl.show()

# %%
# Slice ImageData With Indexing
# +++++++++++++++++++++++++++++
# Most slicing filters return :class:`~pyvista.PolyData` or
# :class:`~pyvista.UnstructuredGrid`. For :class:`~pyvista.ImageData` inputs, however,
# it's often desirable to return :class:`~pyvista.ImageData`. The
# :meth:`~pyvista.ImageDataFilters.slice_index` filter supports this use case.
#
# Extract a single 2D slice from a 3D segmentation mask and plot it. Here we use
# :func:`~pyvista.examples.examples.load_frog_tissues`.

mask = examples.load_frog_tissues()
sliced = mask.slice_index(k=50)
colored = sliced.color_labels()
colored.plot(cpos='xy', zoom='tight', lighting=False)

# %%
# Extract a 3D volume of interest instead and visualize it as a surface mesh. Here we
# define indices to extract the frog's head.

sliced = mask.slice_index(i=[300, 500], j=[110, 350], k=[0, 100])
surface = sliced.contour_labels()
colored = surface.color_labels()

cpos = pv.CameraPosition(
    position=(520.0, 461.0, -402.0),
    focal_point=(372.0, 243.0, 52.0),
    viewup=(-0.73, -0.50, -0.47),
)
colored.plot(cpos=cpos)

# %%
# .. tags:: filter
