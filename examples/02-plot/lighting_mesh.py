"""
.. _lighting_mesh_example:

Lighting Properties
~~~~~~~~~~~~~~~~~~~

Control aspects of the rendered mesh's lighting such as Ambient, Diffuse,
and Specular. These options only work if the ``lighting`` argument to
``add_mesh`` is ``True`` (it's ``True`` by default).

You can turn off all lighting for the given mesh by passing ``lighting=False``
to ``add_mesh``.
"""

# sphinx_gallery_thumbnail_number = 4
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

import pyvista as pv
from pyvista import examples

mesh = examples.download_st_helens().warp_by_scalar()

cpos = pv.CameraPosition(
    position=(575848.0, 5128459.0, 22289.0),
    focal_point=(562835.0, 5114981.5, 2294.5),
    viewup=(-0.5, -0.5, 0.7),
)

# %%
# First, lets take a look at the mesh with default lighting conditions
mesh.plot(cpos=cpos, show_scalar_bar=False)

# %%
# What about with no lighting
mesh.plot(lighting=False, cpos=cpos, show_scalar_bar=False)

# %%
# Demonstration of the specular property

# sphinx_gallery_start_ignore
# specular does not seem to work correctly
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter(shape=(1, 2), window_size=[1500, 500])

pl.subplot(0, 0)
pl.add_mesh(mesh, show_scalar_bar=False)
pl.add_text('No Specular')

pl.subplot(0, 1)
s = 1.0
pl.add_mesh(mesh, specular=s, show_scalar_bar=False)
pl.add_text(f'Specular of {s}')

pl.link_views()
pl.view_isometric()
pl.show(cpos=cpos)

# %%
# Just specular

# sphinx_gallery_start_ignore
# specular does not seem to work correctly
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

mesh.plot(specular=0.5, cpos=cpos, show_scalar_bar=False)

# %%
# Specular power

# sphinx_gallery_start_ignore
# specular does not seem to work correctly
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

mesh.plot(specular=0.5, specular_power=15, cpos=cpos, show_scalar_bar=False)

# %%
# Demonstration of all three in use

# sphinx_gallery_start_ignore
# specular does not seem to work correctly
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

mesh.plot(diffuse=0.5, specular=0.5, ambient=0.5, cpos=cpos, show_scalar_bar=False)

# %%
# For detailed control over lighting conditions in general see the
# :ref:`light_examples` examples.
#
# .. tags:: plot
