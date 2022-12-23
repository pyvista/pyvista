"""
Visible Vertices
~~~~~~~~~~~~~~~~

Display vertices on a mesh in the same fashion as edge visibility.
"""
# sphinx_gallery_thumbnail_number = 3
import pyvista as pv
from pyvista import examples

mesh = examples.download_bunny_coarse()

cpos = [
    (0.036, 0.367, 0.884),
    (0.024, 0.033, -0.022),
    (-0.303, 0.895, -0.325)
]

###############################################################################
# We can easily display all vertices of a mesh with a `points` style
# representation when plotting:
p = pv.Plotter()
p.add_mesh(mesh, style='points', color='magenta', render_points_as_spheres=True, point_size=10)
p.show(cpos=cpos)

###############################################################################
# However, we often want to see the vertices of a mesh rendered atop the
# surface geometry. Much like how we can render the edges of a mesh:
p = pv.Plotter()
p.add_mesh(mesh, show_edges=True)
p.show(cpos=cpos)


###############################################################################
# Displaying the vertices on the surface of a mesh is not as well supported as
# edge visibility. When rendering the edges of a mesh, you'll notice that
# only the surface edges are displayed -- this prevents rendering large
# amounts of internal edges. We have to do the same when displaying the
# surface vertices.
#
# In order to display the vertices atop a mesh's surface geometry, we must
# extract the vertices and display them separately.
#
# The first step is to extract the outer surface geometry of the mesh then
# grab all the points of that extraction.

# Extract surface vertices
nodes = mesh.extract_surface().points

###############################################################################
# Now that we have the vertices extracted, we can use `add_points` to render
# them along side the original geometry.

p = pv.Plotter()
p.add_mesh(mesh, show_edges=True)
p.add_points(nodes, color='magenta', render_points_as_spheres=True, point_size=10)
p.show(cpos=cpos)
