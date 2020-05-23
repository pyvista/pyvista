"""
Clearing a Mesh or the Entire Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to remove elements from a scene.

"""

# sphinx_gallery_thumbnail_number = 3
import pyvista as pv

###############################################################################

plotter = pv.Plotter()
actor = plotter.add_mesh(pv.Sphere())
plotter.remove_actor(actor)
plotter.show()


###############################################################################
# Clearing the entire plotting window:

plotter = pv.Plotter()
plotter.add_mesh(pv.Sphere())
plotter.add_mesh(pv.Plane())
plotter.clear()  # clears all actors
plotter.show()


###############################################################################
# Or you can give any actor a ``name`` when adding it and if an actor is added
# with that same name at a later time, it will replace the previous actor:

plotter = pv.Plotter()
plotter.add_mesh(pv.Sphere(), name="mymesh")
plotter.add_mesh(pv.Plane(), name="mymesh")
# Only the Plane is shown!
plotter.show()
