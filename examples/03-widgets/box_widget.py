"""
.. _box_widget_example:

Box Widget
~~~~~~~~~~

The box widget can be enabled and disabled by the
:func:`pyvista.Plotter.add_box_widget` and
:func:`pyvista.Plotter.clear_box_widgets` methods respectively.
When enabling the box widget, you must provide a custom callback function
otherwise the box would appear and do nothing - the callback functions are
what allow us to leverage the widget to perform a task like clipping/cropping.

Considering that using a box to clip/crop a mesh is one of the most common use
cases, we have included a helper method that will allow you to add a mesh to a
scene with a box widget that controls its extent, the
:func:`pyvista.Plotter.add_mesh_clip_box` method.

.. image:: ../../images/gifs/box-clip.gif
"""

# sphinx_gallery_start_ignore
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

# widgets do not work in interactive examples
from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore


import pyvista as pv
from pyvista import examples

mesh = examples.download_nefertiti()

# %%

pl = pv.Plotter()
pl.add_mesh_clip_box(mesh, color='white')
pl.show(cpos=[-1, -1, 0.2])


# %%
# After interacting with the scene, the clipped mesh is available as:
pl.box_clipped_meshes
# %%
# .. tags:: widgets
