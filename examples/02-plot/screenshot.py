"""
.. _screenshot_example:

Saving Screenshots
~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

# Get a sample file
filename = examples.planefile
mesh = pv.read(filename)


# %%
# You can also take a screenshot without creating an interactive plot window
# using the :class:`pyvista.Plotter`:

pl = pv.Plotter(off_screen=True)
pl.add_mesh(mesh, color='orange')
pl.show(screenshot='airplane.png')

# %%
# The ``img`` array can be used to plot the screenshot in ``matplotlib``:

plt.imshow(pl.image)
plt.show()
# %%
# .. tags:: plot
