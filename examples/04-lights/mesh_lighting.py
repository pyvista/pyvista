"""
.. _mesh_lighting_example:

Disabling Mesh Lighting
~~~~~~~~~~~~~~~~~~~~~~~

Disable mesh lighting.

While plotters have a default set of lights and there are many options
for customizing lighting conditions in general, meshes have the option
to opt out of lighting altogether. Pass ``lighting=False`` to
:func:`pyvista.Plotter.add_mesh` to disable lighting for the given
mesh:

"""

# sphinx_gallery_thumbnail_number = 1
from __future__ import annotations

import pyvista as pv
from pyvista import examples

horse = examples.download_horse().decimate(0.9)
horse.rotate_z(-120, inplace=True)
horse.points = (horse.points - horse.center) * 100
shifted = horse.translate((0, 10, 0), inplace=False)

pl = pv.Plotter()
pl.add_mesh(horse, color='brown')
pl.add_mesh(shifted, color='brown', show_edges=True, lighting=False)
pl.show()


# %%
# Due to the obvious lack of depth detail this mostly makes sense for meshes
# with non-trivial colors or textures. If it weren't for the edges being drawn,
# the second mesh would be practically impossible to understand even with the
# option to interactively explore the surface:

shifted.plot(color='brown', lighting=False)

# %%
# For further examples about fine-tuning mesh properties that affect
# light rendering, see the :ref:`lighting_mesh_example` example.
#
# .. tags:: lights
