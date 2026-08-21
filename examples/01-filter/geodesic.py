"""
.. _geodesic_example:

Geodesic Paths
~~~~~~~~~~~~~~

Calculates the geodesic path between two vertices using Dijkstra's algorithm
"""

# sphinx_gallery_thumbnail_number = 1
import pyvista as pv
from pyvista import examples

# Load a global topography surface and decimate it
land = examples.download_topo_land().triangulate().decimate(0.98)

# %%
# Get the geodesic path as a new :class:`pyvista.PolyData` object:
cape_town = land.find_closest_point((0.790801, 0.264598, -0.551942))
dubai = land.find_closest_point((0.512642, 0.745898, 0.425255))
bangkok = land.find_closest_point((-0.177077, 0.955419, 0.236273))
rome = land.find_closest_point((0.718047, 0.163038, 0.676684))

a = land.geodesic(cape_town, dubai)
b = land.geodesic(cape_town, bangkok)
c = land.geodesic(cape_town, rome)

# %%
# Render the path along the land surface

pl = pv.Plotter()
pl.add_mesh(a + b + c, line_width=10, color='red', label='Geodesic Path')
pl.add_mesh(land, show_edges=True)
pl.add_legend()
pl.camera_position = pv.CameraPosition(
    position=(3.584, 2.392, 1.399),
    focal_point=(-0.06843, 0.1547, -0.07332),
    viewup=(-0.3485, -0.04724, 0.9361),
)

pl.show()

# %%
# How long is that path?
distance = land.geodesic_distance(cape_town, rome)
distance
# %%
# .. tags:: filter
