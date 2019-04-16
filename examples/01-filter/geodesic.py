"""
Geodesic Paths
~~~~~~~~~~~~~~

Calculates the geodesic path betweeen two vertices using Dijkstra's algorithm
"""
# sphinx_gallery_thumbnail_number = 1
import vtki
from vtki import examples

sphere = examples.load_globe()

###############################################################################
# Get teh geodesic path as a new :class:`vtki.PolyData` object:

geodesic = sphere.geodesic(0, sphere.n_points - 1)

###############################################################################
# Render the path along the sphere

p = vtki.Plotter()
p.add_mesh(geodesic, line_width=10, color='red', label='Geodesic Path')
p.add_mesh(sphere, show_edges=True, )
p.camera_position = [-1,-1,1]
p.add_legend()
p.show()

###############################################################################
# How long is that path?
distance = sphere.geodesic_distance(0, sphere.n_points - 1)
print(distance)
