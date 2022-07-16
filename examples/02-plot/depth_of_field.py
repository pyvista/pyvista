"""
.. _depth_of_field_example:

Depth of Field Plotting and Blur
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how you can use :func:`enable_depth_of_field
<pyvista.Plotting.enable_depth_of_field>` to highlight part of your plot.

"""

import numpy as np

import pyvista
from pyvista import examples

###############################################################################
# Generate a bunch of bunnies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create many bunnies using the :func:`glyph <pyvista.DataSetFilters.glyph>`
# filter.

mesh = examples.download_bunny()
mesh = mesh.rotate_x(90, inplace=False).rotate_z(90, inplace=False).scale(4, 4, 4)

grid = pyvista.UniformGrid(dims=(3, 3, 3), spacing=(4, 1, 1))

bunnies = grid.glyph(geom=mesh, scale=False, orient=False)
bunnies

###############################################################################
# Show the plot without enabling depth of field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# convert points into rgba colors
colors = bunnies.points - bunnies.points.min(axis=0)
colors /= colors.max(axis=0)
colors *= 255
colors = colors.astype(np.uint8)

# obtained camera position with `cpos = pl.show(return_cpos)`
cpos = None

pl = pyvista.Plotter()
pl.add_mesh(bunnies, scalars=colors, rgb=True, pbr=True, metallic=0.85)
pl.camera.zoom(1.5)
pl.camera_position = cpos
pl.show()

###############################################################################
# Show the plot while enabling depth of field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pl = pyvista.Plotter()
pl.add_mesh(bunnies, scalars=colors, rgb=True, pbr=True, metallic=0.85)
pl.enable_depth_of_field()
pl.camera_position = cpos
cpos = pl.show(return_cpos=True)
