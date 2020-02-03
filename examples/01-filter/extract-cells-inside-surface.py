"""
Extract Cells Inside Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract the cells in a mesh that exist inside or outside a closed surface of
another mesh

"""
###############################################################################

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

mesh = examples.download_cow()

cpos = [(13.0, 7.6, -13.85), (0.44, -0.4, -0.37), (-0.28, 0.9, 0.3)]

dargs = dict(show_edges=True)
# Rotate the mesh to have a second mesh
rot = mesh.copy()
rot.rotate_y(90)

p = pv.Plotter()
p.add_mesh(mesh, color="Crimson", **dargs)
p.add_mesh(rot, color="mintcream", opacity=0.35, **dargs)
p.camera_position = cpos
p.show()

###############################################################################
# Mark points inside with 1 and outside with a 0
select = mesh.select_enclosed_points(rot)

select
###############################################################################
# Extract two meshes, one completely inside and one completely outside the
# enclosing surface.

inside = select.threshold(0.5)
outside = select.threshold(0.5, invert=True)

###############################################################################
# display the results

p = pv.Plotter()
p.add_mesh(outside, color="Crimson", **dargs)
p.add_mesh(inside, color="green", **dargs)
p.add_mesh(rot, color="mintcream", opacity=0.35, **dargs)

p.camera_position = cpos
p.show()
