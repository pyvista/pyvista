"""
.. _depth_of_field_example:

Depth of Field Plotting
~~~~~~~~~~~~~~~~~~~~~~~

This example shows how you can use :func:`enable_depth_of_field
<pyvista.Plotter.enable_depth_of_field>` to highlight part of your plot.

"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# depth field modification does not seem to work in interactive mode
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import numpy as np
import pyvista as pv
from pyvista import examples

# %%
# Generate a bunch of bunnies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create many bunnies using the :func:`glyph <pyvista.DataSetFilters.glyph>`
# filter.

# download the stanford bunny and rotate it into a good position
mesh = examples.download_bunny()
mesh = mesh.rotate_x(90, inplace=False).rotate_z(90, inplace=False).scale(4)

# We use a uniform grid here simply to create equidistantly spaced points for
# our glyph filter
grid = pv.ImageData(dimensions=(4, 3, 3), spacing=(3, 1, 1))

bunnies = grid.glyph(geom=mesh, scale=False, orient=False)
bunnies


# %%
# Show the plot without enabling depth of field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# convert points into rgba colors
colors = bunnies.points - bunnies.bounds[::2]
colors /= colors.max(axis=0)
colors *= 255
colors = colors.astype(np.uint8)

# obtained camera position with `cpos = pl.show(return_cpos)`
cpos = pv.CameraPosition(
    position=(11.6159, -1.2803, 1.5338),
    focal_point=(4.1354, 1.4796, 1.2711),
    viewup=(-0.0352, -0.0004, 1.0),
)

# Since we're using physically based rendering (PBR), let's also download a
# skybox cubemap use it as an environment texture. For PBR to work well you
# should have a environment texture.
cubemap = examples.download_sky_box_cube_map()

pl = pv.Plotter()
pl.background_color = 'w'
pl.add_mesh(bunnies, scalars=colors, rgb=True, pbr=True, metallic=0.85)
pl.camera_position = cpos
pl.set_environment_texture(cubemap)
pl.show()


# %%
# Show the plot while enabling depth of field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pl = pv.Plotter()
pl.background_color = 'w'
pl.add_mesh(bunnies, scalars=colors, rgb=True, pbr=True, metallic=0.85)
pl.camera_position = cpos
pl.enable_depth_of_field()
pl.enable_anti_aliasing('ssaa')
pl.set_environment_texture(cubemap)
pl.show()
# %%
# .. tags:: plot
