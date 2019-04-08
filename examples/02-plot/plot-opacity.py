"""
Plot with Opacity
~~~~~~~~~~~~~~~~~

Plot a mesh's scalar array with an opacity trasfer funciton
"""

import vtki
from vtki import examples

################################################################################
# It's possible to apply an opacity mapping to any scalar array plotted. You can
# specify either a single static value to make the mesh opaque on all cells, or
# use a transfer function where the scalar array plotted is mapped to the opacity.
#
# Opacity transfer function options are:
#
# - ``'linear'``: linearly vary (increase) opacity across the plotted scalar range from low to high
# - ``'linear_r'``: linearly vary (increase) opacity across the plotted scalar range from high to low
# - ``'geom'``: on a log scale, vary (increase) opacity across the plotted scalar range from low to high
# - ``'geom_r'``: on a log scale, vary (increase) opacity across the plotted scalar range from high to low

# Load St Helens DEM and warp the topography
mesh = examples.download_st_helens().warp_by_scalar()

mesh.plot(opacity='linear')


################################################################################
# Opacity mapping is often useful when plotting DICOM images. For example,
# download the sample knee DICOM image:
knee = examples.download_knee()

################################################################################
# And here we inspect the DICOM image with a few different colormaps and opacity
# mappings:
p = vtki.Plotter(shape=(2,2), border=False)

p.add_mesh(knee, cmap='bone', show_scalar_bar=0)
p.view_xy()

p.subplot(0,1)
p.add_mesh(knee, cmap='bone', opacity='linear', show_scalar_bar=0)
p.view_xy()

p.subplot(1,0)
p.add_mesh(knee, cmap='nipy_spectral', show_scalar_bar=0)
p.view_xy()

p.subplot(1,1)
p.add_mesh(knee, cmap='nipy_spectral', opacity='linear', show_scalar_bar=0)
p.view_xy()

p.show()
