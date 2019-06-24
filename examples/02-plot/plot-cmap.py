"""
Custom Colormaps
~~~~~~~~~~~~~~~~

Use a custom built colormap when plotting scalar values.
"""

from pyvista import examples
import pyvista as pv
import matplotlib.pyplot as plt

################################################################################
# Any colormap built for ``matplotlib`` or ``colorcet`` is fully compatible
# with PyVista. Colormaps are typically specified by passing the string name of
# the ``matplotlib`` or ``colorcet`` colormap to the plotting routine via the
# ``cmap`` argument.
#
# See `Matplotlib's complete list of available colormaps`_ and
# `Colorcet's complete list`_.
#
# .. _Matplotlib's complete list of available colormaps: https://matplotlib.org/tutorials/colors/colormaps.html
# .. _Colorcet's complete list: http://colorcet.pyviz.org/user_guide/index.html
#
# To get started using a custom colormap, download some data with scalars to
# plot.

mesh = examples.download_st_helens().warp_by_scalar()

################################################################################
# Build a custom colormap - here we just make a viridis map with 5 discrete
# colors, but you could make this as complex or simple as you desire.

cmap = plt.cm.get_cmap('viridis', 5)

################################################################################
# Simply pass the colormap to the plotting routine!
mesh.plot(cmap=cmap, cpos='xy')


################################################################################
# Matplotlib vs. Colorcet
# +++++++++++++++++++++++
#
# Let's compare Colorcet's perceptually uniform "fire" colormap to Matplotlib's
# "hot" colormap much like the example on the `first page of Colorcet's docs`_.
#
# .. _first page of Colorcet's docs: http://colorcet.pyviz.org/index.html
#
# The "hot" version washes out detail at the high end, as if the image is
# overexposed, while "fire" makes detail visible throughout the data range.

p = pv.Plotter(shape=(2,2), border=False)
p.subplot(0,0)
p.add_mesh(mesh, cmap='fire', lighting=True, stitle='Colorcet Fire')

p.subplot(0,1)
p.add_mesh(mesh, cmap='fire', lighting=False, stitle='Colorcet Fire (No Lighting)')

p.subplot(1,0)
p.add_mesh(mesh, cmap='hot', lighting=True, stitle='Matplotlib Hot')

p.subplot(1,1)
p.add_mesh(mesh, cmap='hot', lighting=False, stitle='Matplotlib Hot (No Lighting)')

p.show()
