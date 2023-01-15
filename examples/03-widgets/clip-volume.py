"""
.. _clip_volume_widget_example:

Clip Volume Widget
~~~~~~~~~~~~~~~~~~
If you have a :class:`pyvista.UniformGrid`, you can clip it using the
:func:`pyvista.Plotter.add_volume_clipper` widget.


.. image:: ../../images/gifs/box-clip.gif
"""


import numpy as np

import pyvista as pv
from pyvista import examples

# pv.set_plot_theme('dark')


# Create the dataset (hydrogen 3d orbital with m=0)
grid = examples.load_hydrogen_orbital(3, 2, 0, norm=False)


###############################################################################
# Plot the Probability Density of the Orbital
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pl = pv.Plotter(theme=pv.themes.DarkTheme())
vol = pl.add_volume(grid, opacity='linear', cmap='magma')
vol.prop.interpolation_type = 'linear'
# pl.camera_position ='yz'
pl.add_volume_clipper(vol, normal='-x', implicit=True)
pl.show_axes()
pl.show()


###############################################################################
# Plot the 3d Orbital Contours
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

contours = grid.contour([grid['norm_hwf'].max() * 0.1], method='marching_cubes')
contours = contours.interpolate(grid)
contours.plot(scalars=np.real(contours['hwf']), show_scalar_bar=False, smooth_shading=True)
