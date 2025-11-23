"""
.. _clear_example:

Clearing a Mesh or the Entire Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to remove elements from a scene using
:meth:`~pyvista.Plotter.clear`.

"""

# sphinx_gallery_thumbnail_number = 3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

import pyvista as pv

# %%

pl = pv.Plotter()
actor = pl.add_mesh(pv.Sphere())
pl.remove_actor(actor)
pl.show()


# %%
# Clearing the entire plotting window:

pl = pv.Plotter()
pl.add_mesh(pv.Sphere())
pl.add_mesh(pv.Plane())
pl.clear()  # clears all actors
pl.show()


# %%
# Or you can give any actor a ``name`` when adding it and if an actor is added
# with that same name at a later time, it will replace the previous actor:

pl = pv.Plotter()
pl.add_mesh(pv.Sphere(), name='mymesh')
pl.add_mesh(pv.Plane(), name='mymesh')
# Only the Plane is shown.
pl.show()
# %%
# .. tags:: plot
