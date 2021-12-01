"""
.. _point_picking_example:

Picking points on a mesh
~~~~~~~~~~~~~~~~~~~~~~~~

"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv

###############################################################################

sphere = pv.Sphere()

p = pv.Plotter()
sphere_actor = p.add_mesh(sphere, pickable=False)

p.enable_point_picking(pickable_window=False)
sphere_actor.SetPickable(True)
p.show()
