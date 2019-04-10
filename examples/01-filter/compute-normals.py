"""
Computing Surface Normals
~~~~~~~~~~~~~~~~~~~~~~~~~


Compute normals on a surface.
"""

# sphinx_gallery_thumbnail_number = 2
import vtki
from vtki import examples

################################################################################
# Computing the normals of a surace is quite easy using :class:`vtki.PolyData`'s
# ``compute_normals`` method

mesh = examples.download_topo_global()
mesh.plot(cmap='gist_earth', show_scalar_bar=False)

################################################################################
# Now we have a surface dataset of the globe loaded - unfortunately, the dataset
# shows the globe with a uniform radius which hides topographic releif.
# Using :func:`vtki.PolyData.compute_normals`, we can compute the normal vectors
# on the globe at all points in the dataset, then use the values given in the
# dataset to warp the surface in the normals direction to create some
# exaggerated topographic relief.

# Compute the normals in-place and use them to warp the globe
mesh.compute_normals() # this activates the normals as well

# Now use those normals to warp the surface
warp = mesh.warp_by_scalar(scale_factor=0.5e-5)

# And let's see it!
warp.plot(cmap='gist_earth', show_scalar_bar=False)
