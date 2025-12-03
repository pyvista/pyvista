"""
.. _moving_isovalue_example:

Moving Isovalue
~~~~~~~~~~~~~~~

Make an animation of an isovalue through a volumetric dataset
such as :func:`~pyvista.examples.downloads.download_brain`.
This example uses :meth:`~pyvista.Plotter.open_gif` and
:meth:`~pyvista.Plotter.write_frame` to create the animation.

"""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples

vol = examples.download_brain()
vol

# %%
# Now lets make an array of all of the isovalues for which we want to show.
values = np.linspace(5, 150, num=25)

# %%
# Now let's create an initial isosurface that we can plot and move
surface = vol.contour(values[:1])

# %%
# Precompute the surfaces
surfaces = [vol.contour([v]) for v in values]

# %%
# Set a single surface as the one being plotted that can be overwritten
surface = surfaces[0].copy()

# %%

filename = 'isovalue.gif'

pl = pv.Plotter(off_screen=True)
# Open a movie file
pl.open_gif(filename)

# Add initial mesh
pl.add_mesh(
    surface,
    opacity=0.5,
    clim=vol.get_data_range(),
    show_scalar_bar=False,
)
# Add outline for reference
pl.add_mesh(vol.outline_corners(), color='k')

print('Orient the view, then press "q" to close window and produce movie')
pl.camera_position = pv.CameraPosition(
    position=(392.9783280407326, 556.4341372317185, 235.51220650196404),
    focal_point=(88.69563012828344, 119.06774369173661, 72.61750326143748),
    viewup=(-0.19275936948097383, -0.2218876327549124, 0.9558293278131397),
)

# initial render and do NOT close
pl.show(auto_close=False)

# Run through each frame
for surf in surfaces:
    surface.copy_from(surf)
    pl.write_frame()  # Write this frame
# Run through backwards
for surf in surfaces[::-1]:
    surface.copy_from(surf)
    pl.write_frame()  # Write this frame

# Be sure to close the plotter when finished
pl.close()
# %%
# .. tags:: plot
