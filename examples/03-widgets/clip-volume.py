"""
.. _clip_volume_widget_example:

Clip Volume Widget
------------------
If you have a structured dataset like :class:`pyvista.UniformGrid`, or
:class:`pyvista.RectilinearGrid` you can clip it using the
:func:`pyvista.Plotter.add_volume_clipper` widget.  .. image::

../../images/gifs/box-clip.gif

"""

import numpy as np

import pyvista as pv

###############################################################################
# Create the Dataset
# ~~~~~~~~~~~~~~~~~~
# Create a dense :class:`pyvista.UniformGrid` with dimensions ``(200, 200,
# 200)`` and set the active scalars to distance from the :attr:`center
# <pyvista.UniformGrid.center>` of the grid.

grid = pv.UniformGrid(dimensions=(200, 200, 200))

scalars = np.linalg.norm(grid.center - grid.points, axis=1)
# scalars = np.abs(scalars.max() - scalars)
grid['scalars'] = scalars

opacity = np.zeros(100)
opacity[::10] = np.geomspace(0.01, 0.5, 10)
x = np.linspace(0, 1, len(opacity))


pl = pv.Plotter()
vol = pl.add_volume(grid, opacity=opacity)
vol.prop.interpolation_type = 'linear'
pl.add_volume_clip_plane(vol, event_type='always')
pl.show()
