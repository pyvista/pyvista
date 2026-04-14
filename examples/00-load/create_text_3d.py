"""
.. _create_text_3d_example:

Create 3D Text
~~~~~~~~~~~~~~

Generate extruded text geometry with :func:`pyvista.Text3D`.
"""

from __future__ import annotations

import pyvista as pv

# %%
# Create an extruded text mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The text is standard PolyData, so it can be transformed and plotted like any
# other surface.

text = pv.Text3D('PyVista', depth=0.3)
text


# %%
# Compare flat and extruded text
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setting ``depth=0`` creates a flat version while a positive depth extrudes the
# glyphs into 3D.

flat = pv.Text3D('PyVista', depth=0)
extruded = pv.Text3D('PyVista', depth=0.4)

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_mesh(flat, color='royalblue')
pl.subplot(0, 1)
pl.add_mesh(extruded, color='royalblue')
pl.link_views()
pl.show()
# %%
# .. tags:: load
