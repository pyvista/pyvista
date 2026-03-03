"""
.. _clip_volume_widget_example:

Clip Volume Widget
------------------
If you have a structured dataset like a :class:`pyvista.ImageData` or
:class:`pyvista.RectilinearGrid`, you can clip it using the
:func:`pyvista.Plotter.add_volume_clip_plane` widget to better see the internal
structure of the dataset.

.. image:: ../../images/gifs/volume-clip-plane-widget.gif

"""

# sphinx_gallery_start_ignore
# widgets do not work in interactive examples
from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Create the Dataset
# ~~~~~~~~~~~~~~~~~~
# Create a dense :class:`pyvista.ImageData` with dimensions ``(200, 200,
# 200)`` and set the active scalars to distance from the :attr:`center
# <pyvista.DataSet.center>` of the grid.

import numpy as np
import pyvista as pv

grid = pv.ImageData(dimensions=(200, 200, 200))
grid['scalars'] = np.linalg.norm(grid.center - grid.points, axis=1)
grid


# %%
# Generate the Opacity Array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a banded opacity array such that our dataset shows "rings" at certain
# values. Have this increase such that higher values (values farther away from
# the center) are more opaque.

opacity = np.zeros(100)
opacity[::10] = np.geomspace(0.01, 0.75, 10)


# %%
# Plot a Single Clip Plane Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the volume with a single clip plane.
#
# Reverse the opacity array such that portions closer to the center are more
# opaque.

pl = pv.Plotter()
pl.add_volume_clip_plane(grid, normal='-x', opacity=opacity[::-1], cmap='magma')
pl.show()


# %%
# Plot Multiple Clip Planes
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the dataset using the :func:`pyvista.Plotter.add_volume_clip_plane` with
# the output from :func:`pyvista.Plotter.add_volume` Enable constant
# interaction by setting the ``interaction_event`` to ``'always'``.
#
# Disable the arrows to make the plot a bit clearer and flip the opacity array.

pl = pv.Plotter()
vol = pl.add_volume(grid, opacity=opacity)
vol.prop.interpolation_type = 'linear'
pl.add_volume_clip_plane(
    vol,
    normal='-x',
    interaction_event='always',
    normal_rotation=False,
)
pl.add_volume_clip_plane(
    vol,
    normal='-y',
    interaction_event='always',
    normal_rotation=False,
)
pl.show()
# %%
# .. tags:: widgets
