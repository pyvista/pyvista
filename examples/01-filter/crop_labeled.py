"""
.. _crop_labeled_example:

Crop Labeled ImageData
~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`~pyvista.ImageDataFilters.crop` to crop labeled data such as segmented medical images.

"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# %%
# Load a dataset with a CT image and segmentation labels. Here we load
# :func:`~pyvista.examples.downloads.download_whole_body_ct_male`.

dataset = examples.download_whole_body_ct_male()

# %%
# Get the :class:`~pyvista.ImageData` for the CT data and one of the segmentation masks.
# For this example we choose a mask of the skull.

ct = dataset['ct']
skull = dataset['segmentations']['skull']

# %%
# Crop the CT image using the segmentation mask. Use ``padding`` to include additional data
# points around the masked region in the output.

cropped_ct = ct.crop(mask=skull, padding=10)

# %%
# Use :meth:`~pyvista.ImageDataFilters.points_to_cells` to plot the cropped image
# as :attr:`~pyvista.CellType.VOXEL` cells.

cpos = [(687.5, 763.6, 471.3), (231.8, 296.3, 677.0), (0.107, 0.311, 0.944)]

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
