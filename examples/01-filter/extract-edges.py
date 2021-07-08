"""
.. _extract_edges_example:

Extract Edges
~~~~~~~~~~~~~

Extracts edges from a surface.
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

###############################################################################
# From vtk documentation, the edges of a mesh are one of the following:
#
# 1. boundary (used by one polygon) or a line cell
# 2. non-manifold (used by three or more polygons)
# 3. feature edges (edges used by two triangles and whose dihedral angle > feature_angle)
# 4. manifold edges (edges used by exactly two polygons).
#
# This filter will extract those edges given a feature angle and return a dataset
# with lines that represent the edges of the original mesh.
# To demonstrate, we will first extract the edges around Queen Nefertiti's eyes:

# Load Queen Nefertiti mesh
mesh = examples.download_nefertiti()

# Extract the edges above a 12 degree feature angle
edges = mesh.extract_feature_edges(12)

# Render the edge lines on top of the original mesh
p = pv.Plotter()
p.add_mesh(mesh, color=True)
p.add_mesh(edges, color="red", line_width=5)
# Define a camera position that will zoom to her eye
p.camera_position = [(96.0, -197.0, 45.0), (7.0, -109.0, 22.0), (0, 0, 1)]
p.show()

###############################################################################
# We can do this analysis for any :class:`pyvista.PolyData` object. Let's try
# the cow mesh example:

mesh = examples.download_cow()

edges = mesh.extract_feature_edges(20)

p = pv.Plotter()
p.add_mesh(mesh, color=True)
p.add_mesh(edges, color="red", line_width=5)
p.camera_position = [(9.5, 3.0, 5.5), (2.5, 1, 0), (0, 1, 0)]
p.show()


###############################################################################
# We can leverage the :any:`pyvista.PolyData.n_open_edges` property and
# :func:`pyvista.PolyDataFilters.extract_feature_edges` filter to count and extract the
# open edges on a :class:`pyvista.PolyData` mesh.

# Download a sample surface mesh with visible open edges
mesh = examples.download_bunny()

###############################################################################
# We can get a count of the open edges with:
mesh.n_open_edges


###############################################################################
# And we can extract those edges with the ``boundary_edges`` option of
# :func:`pyvista.PolyDataFilters.extract_feature_edges`:
edges = mesh.extract_feature_edges(boundary_edges=True,
                           feature_edges=False,
                           manifold_edges=False)

p = pv.Plotter()
p.add_mesh(mesh, color=True)
p.add_mesh(edges, color="red", line_width=5)
p.camera_position = [(-0.2, -0.13, 0.12), (-0.015, 0.10, -0.0), (0.28, 0.26, 0.9)]
p.show()
