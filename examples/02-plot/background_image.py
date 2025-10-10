"""
.. _background_image_example:

Background Image
~~~~~~~~~~~~~~~~

Add a background image with :func:`pyvista.Plotter.add_background_image`.

"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# sphinx_gallery_start_ignore
# background images do not seem to work in interactive mode
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Plot an airplane with the map of the earth in the background
earth_alt = examples.download_topo_global()

pl = pv.Plotter()
actor = pl.add_mesh(examples.load_airplane(), smooth_shading=True)
pl.add_background_image(examples.mapfile)
pl.show()

# %%
# Plot several earth related plots

pl = pv.Plotter(shape=(2, 2))

pl.subplot(0, 0)
pl.add_text('Earth Visible as Map')
pl.add_background_image(examples.mapfile, as_global=False)

pl.subplot(0, 1)
pl.add_text('Earth Altitude')
actor = pl.add_mesh(earth_alt, cmap='gist_earth')

pl.subplot(1, 0)
topo = examples.download_topo_land()
actor = pl.add_mesh(topo, cmap='gist_earth')
pl.add_text('Earth Land Altitude')

pl.subplot(1, 1)
globe = examples.load_globe()
texture = examples.load_globe_texture()
pl.add_text('Earth Visible as Globe')
pl.add_mesh(globe, texture=texture, smooth_shading=True)

pl.show()
# %%
# .. tags:: plot
