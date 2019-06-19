"""
Slicing
~~~~~~~

Extract thin planar slices from a volume
"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# PyVista meshes have several slicing filters bound directly to all datasets.
# Thes filters allow you to slice through a volumetric dataset to extract and
# view sections through the volume of data.
#
# One of the most common slicing filters used in PyVista is the
# :func:`pyvista.DataSetFilters.slice_orthogonal` filter which creates three
# orthogonal slices through the dataset on the three caresian planes.
# For example, let's slice through the sample geostatitical training image
# volume. First, load up the volume and preview it:

mesh = examples.load_channels()
# define a categorical colormap
cmap = plt.cm.get_cmap('viridis', 4)


mesh.plot(cmap=cmap)

################################################################################
# Note that this dataset is a 3D volume and their might be regions within the
# volume that we would like to inspect. We can create slices through the mesh
# to gain insight about the internals of the volume.

slices = mesh.slice_orthogonal()

slices.plot(cmap=cmap)


################################################################################
# The orthogonal slices can be easily translated throughout the volume:

slices = mesh.slice_orthogonal(x=20, y=20, z=30)
slices.plot(cmap=cmap)
################################################################################
# We can also add just a single slice of the volume by specifying the origin
# and normal of the slicing plane with the :func:`pyvista.DataSetFilters.slice`
# filter:

# Sing slice - origin defaults to center of the mesh
single_slice = mesh.slice(normal=[1,1,0])

p = pv.Plotter()
p.add_mesh(mesh.outline(), color='k')
p.add_mesh(single_slice, cmap=cmap)
p.show()
################################################################################
# Adding slicing planes uniformly across an axial direction can also be
# automated with the :func:`pyvista.DataSetFilters.slice_along_axis` filter:

slices = mesh.slice_along_axis(n=7, axis='y')

slices.plot(cmap=cmap)


################################################################################
# Slice Along Poly Line
# +++++++++++++++++++++
#
# We can also slice a dataset along a :func:`pyvista.Spline` or ``PolyLine``
# contained in :class:`pyvista.PolyData` using the
# :func:`DataSetFilters.slice_along_line` filter.
#
# First, define a spline/polyline through a datset of interest. Please note
# that this type of slicing is computationally expensive and might take a while
# if there are a lot of points in the spline - try to keep the resolution of
# the spline low.

model = examples.load_channels()

def path(y):
    """Equation: x = a(y-h)^2 + k"""
    a = 110.0 / 160.0**2
    x = a*y**2 + 0.0
    return x, y

x, y = path(np.arange(model.bounds[2], model.bounds[3], 15.0))
zo = np.linspace(9.0, 11.0, num=len(y))
points = np.c_[x,y,zo]
spline = pv.Spline(points, 15)
print(spline)


################################################################################
# Then run the filter
slc = model.slice_along_line(spline)
print(slc)

################################################################################

p = pv.Plotter()
p.add_mesh(slc)
p.add_mesh(model.outline())
p.show(cpos=[1,-1,1])
