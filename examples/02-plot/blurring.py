"""
.. _blurring_example:

Blurring
~~~~~~~~

Blur a plot, or highlight part of it, using depth-of-field style effects.

Uses :func:`add_blurring <pyvista.Plotter.add_blurring>` or
:func:`enable_depth_of_field <pyvista.Plotter.enable_depth_of_field>`.

"""

import pyvista as pv

# sphinx_gallery_start_ignore
# blurring does not work in interactive examples probably because interactive
# plot resets some properties of the camera.
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Create Several Spheres
# ~~~~~~~~~~~~~~~~~~~~~~

# We use a uniform grid here simply to create equidistantly spaced points for
# our glyph filter
grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(1, 1, 1))

spheres = grid.glyph(geom=pv.Sphere(), scale=False, orient=False)


# %%
# Blur the Plot
# ~~~~~~~~~~~~~
# Add a few blur passes to blur the plot

pl = pv.Plotter()
pl.add_mesh(spheres, smooth_shading=True, show_edges=True)
pl.add_blurring()
pl.add_blurring()
pl.add_blurring()
pl.camera.zoom(1.5)
pl.enable_anti_aliasing('ssaa')
pl.show()


# %%
# Note how this is different than selectively blurring part of the mesh behind
# the focal plane

pl = pv.Plotter()
pl.add_mesh(spheres, smooth_shading=True, show_edges=True)
pl.enable_depth_of_field()
pl.camera.zoom(1.5)
pl.enable_anti_aliasing('ssaa')
pl.show()
# %%
# .. tags:: plot
