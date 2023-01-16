"""
.. _clip_volume_widget_example:

Clip Volume Widget
~~~~~~~~~~~~~~~~~~
If you have a :class:`pyvista.UniformGrid`, you can clip it using the
:func:`pyvista.Plotter.add_volume_clipper` widget.


.. image:: ../../images/gifs/box-clip.gif
"""


import pyvista as pv
from pyvista import examples

# Create the dataset (hydrogen 3d orbital with n=3, l=2, and m=0)
grid = examples.load_hydrogen_orbital(3, 2, -2)


###############################################################################
# Plot the Probability Density of the Orbital
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pl = pv.Plotter(theme=pv.themes.DarkTheme())
vol = pl.add_volume(grid, cmap='magma', opacity=[1, 0, 1])
vol.prop.interpolation_type = 'linear'
# pl.camera_position ='yz'
pl.add_volume_clipper(vol, normal='-x', implicit=True)
pl.show_axes()
pl.show()


###############################################################################
# Plot the Orbital Contours
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# contours = grid.contour([grid['norm_hwf'].max() * 0.1], method='marching_cubes')
# contours = contours.interpolate(grid)
# contours.plot(scalars=np.real(contours['hwf']), show_scalar_bar=False, smooth_shading=True)
