"""
Clearing a Mesh or the Entire Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to remove elements from a scene.

"""

# sphinx_gallery_thumbnail_number = 3
import pyvista

################################################################################

plotter = pyvista.Plotter()
actor = plotter.add_mesh(pyvista.Sphere())
plotter.remove_actor(actor)
plotter.show()


################################################################################
# Clearing the entire plotting window:

plotter = pyvista.Plotter()
plotter.add_mesh(pyvista.Sphere())
plotter.add_mesh(pyvista.Plane())
plotter.clear()  # clears all actors
plotter.show()


################################################################################
# Or you can give any actor a ``name`` when adding it and if an actor is added
# with that same name at a later time, it will replace the previous actor:

plotter = pyvista.Plotter()
plotter.add_mesh(pyvista.Sphere(), name='mydata')
plotter.add_mesh(pyvista.Plane(), name='mydata')
# Only the Plane is shown!
plotter.show()
