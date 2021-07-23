"""
.. _surface_smoothing_example:

Surface Smoothing
~~~~~~~~~~~~~~~~~

Smoothing rough edges of a surface mesh
"""

# sphinx_gallery_thumbnail_number = 4
from pyvista import examples

###############################################################################
# Suppose you extract a volumetric subset of a dataset that has roughly defined
# edges. Perhaps you'd like a smooth representation of that model region. This
# can be achieved by extracting the bounding surface of the volume and applying
# a :func:`pyvista.PolyData.smooth` filter.
#
# The below code snippet loads a sample roughly edged volumetric dataset:

# Vector to view rough edges
cpos = [-2, 5, 3]

# Load dataset
data = examples.load_uniform()
# Extract a rugged volume
vol = data.threshold_percent(30, invert=1)
vol.plot(show_edges=True, cpos=cpos)

###############################################################################
# Extract the outer surface of the volume using the
# :func:`pyvista.DataSetFilters.extract_geometry` filter and then apply the
# smoothing filter:

# Get the out surface as PolyData
surf = vol.extract_geometry()
# Smooth the surface
smooth = surf.smooth()
smooth.plot(show_edges=True, cpos=cpos)

###############################################################################
# Not smooth enough? Try increasing the number of iterations for the Laplacian
# smoothing algorithm:

# Smooth the surface even more
smooth = surf.smooth(n_iter=100)
smooth.plot(show_edges=True, cpos=cpos)

###############################################################################
# Still not smooth enough? Increase the number of iterations for the Laplacian
# smoothing algorithm to a crazy high value:

# Smooth the surface EVEN MORE
smooth = surf.smooth(n_iter=1000)
smooth.plot(show_edges=True, cpos=cpos)
