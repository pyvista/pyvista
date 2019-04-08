"""
Custom Colormaps
~~~~~~~~~~~~~~~~

Use a custom built colormap when plotting scalar values.
"""

import vtki
from vtki import examples
import matplotlib.pyplot as plt

################################################################################
# Any colormap built for ``matplotlib`` is fully compatible with ``vtki``.
# Colormaps are typically specifiedby passing the string name of the
# ``matplotlib`` colormap to the plotting routine via the ``cmap`` argument.
#
# See `this page`_ for a complete list of available colormaps.
#
# .. _this page: https://matplotlib.org/tutorials/colors/colormaps.html
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
