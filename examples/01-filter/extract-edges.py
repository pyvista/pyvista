"""
Extract Edges
~~~~~~~~~~~~~~~~~

Extracts edges from a surface.
"""

# sphinx_gallery_thumbnail_number = 2
import vtki
from vtki import examples

################################################################################
# From vtk documentation, the edges are one of the following:
#
# 1. boundary (used by one polygon) or a line cell
# 2. non-manifold (used by three or more polygons)
# 3. feature edges (edges used by two triangles and whose dihedral angle > feature_angle)
# 4. manifold edges (edges used by exactly two polygons).
#
# This filter will extract those edges given a feature angle and return a datset
# with lines that represent the edges of the original mesh.
# To demonstrate, we will first extract the edges around Queen Nefertiti's eyes:

# Load Queen Nefertiti mesh
mesh = examples.download_nefertiti()

# Extract the edges above a 12 degree feature angle
edges = mesh.extract_edges(12)

# Render the edge lines ontop of the original mesh
p = vtki.Plotter()
p.add_mesh(mesh, color=True)
p.add_mesh(edges, color='red', line_width=5)
# Define a camera position that will zoom to her eye
p.camera_position = [(96., -197., 45.),
                     (7., -109., 22.),
                     (0, 0, 1)]
p.show()

################################################################################
# We can do this anaylsis for any :class:`vtki.PolyData` object. Let's try the
# cow mesh example:

mesh = examples.download_cow()

edges = mesh.extract_edges(20)

p = vtki.Plotter()
p.add_mesh(mesh, color=True)
p.add_mesh(edges, color='red', line_width=5)
p.camera_position = [(9.5, 3., 5.5),
                     (2.5, 1, 0),
                     (0, 1, 0)]
p.show()
