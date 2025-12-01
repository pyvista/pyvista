"""
.. _crop_labeled_example:

Crop Labeled ImageData
~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`~pyvista.ImageDataFilters.crop` to crop labeled data such as segmented medical images.

"""

# sphinx_gallery_thumbnail_number = 2

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# %%
# Load a dataset with a CT image and corresponding segmentation labels. Here we load
# :func:`~pyvista.examples.downloads.download_whole_body_ct_male`.

dataset = examples.download_whole_body_ct_male()

# %%
# Get the :class:`~pyvista.ImageData` for the CT data and one of the segmentation masks.
# For this example we choose a mask of the skull.

ct = dataset['ct']
skull = dataset['segmentations']['skull']

# %%
# Crop the CT image using the segmentation mask. Use ``padding`` to include additional data
# points around the masked region.

cropped_ct = ct.crop(mask=skull, padding=10)

# %%
# Use :meth:`~pyvista.ImageDataFilters.points_to_cells` to plot the cropped image
# as :attr:`~pyvista.CellType.VOXEL` cells.

cpos = pv.CameraPosition(
    position=(687.5, 763.6, 471.3), focal_point=(231.8, 296.3, 677.0), viewup=(0.107, 0.311, 0.944)
)

cropped_ct_voxels = cropped_ct.points_to_cells()
cropped_ct_voxels.plot(volume=True, cpos=cpos)

# %%
# Include a surface contour of the mask with the plot.

skull_surface = skull.contour_labels()

pl = pv.Plotter()
pl.add_mesh(skull_surface, color='white')
pl.add_volume(cropped_ct_voxels)
pl.camera_position = cpos
pl.show()

# %%
# After cropping, the CT image's dimensions are smaller than the mask's.

cropped_ct.dimensions == skull.dimensions

# %%
# To keep dimension the same, either
#
# #. crop the mask itself; the meshes will have smaller dimensions relative to the input
# #. pad the CT image as part of the initial crop; the meshes will have the same dimensions
#    as the input
#
# To crop the mask itself, you can perform a similar crop as before using ``mask=True``.

cropped_skull = skull.crop(mask=True, padding=10)
cropped_skull.dimensions

# %%
# However, computationally it is more efficient to crop using ``extent`` directly.

cropped_skull = skull.crop(extent=cropped_ct.extent)
cropped_skull.dimensions

# %%
# Alternatively, use ``keep_dimensions`` and ``fill_value`` when initially cropping the image so
# that the output dimensions match the input. A value of ``-1000`` is used, which may represent
# air in the scan.

cropped_ct = ct.crop(mask=skull, keep_dimensions=True, fill_value=-1000)
cropped_ct.dimensions
