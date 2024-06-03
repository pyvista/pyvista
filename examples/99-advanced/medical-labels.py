"""
.. _medical_labels:

Process Medical Segmentation Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example walks through a few processing steps and visualization methods
which are useful for working with medical images and segmentation labels.
Many of these steps are similar to those used to load the datasets
:func:`~pyvista.examples.downloads.download_whole_body_ct_female` and
:func:`~pyvista.examples.downloads.download_whole_body_ct_male`.

"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import pyvista as pv
from pyvista import examples

###############################################################################
# Load Dataset
# ------------
# Load a medical image and its corresponding binary segmentation masks.
#
# For this example, we use data from the TotalSegmentator dataset which can be
# downloaded from `zenodo <https://zenodo.org/records/10047292>`_. Each subject
# from this dataset contains a single CT image and a folder with segmentation
# masks of 117 anatomical structures.
#
# Two subjects from this source, one
# :func:`male <pyvista.examples.downloads.download_whole_body_ct_male>` (``'s1397'``)
# and one
# :func:`female <pyvista.examples.downloads.download_whole_body_ct_female>` (``'s1380'``),
# are available for download in PyVista. In this example, we download the files
# associated with the male dataset. However, this example can be applied to any
# subject from the TotalSegmentator dataset or other medical segmentation data.

###############################################################################
# Load CT Image
# =============
# Download the dataset. Read the NIFTI file of the CT image.
dataset_folder = examples.download_whole_body_ct_male(load=False)
ct_path = str(Path(dataset_folder, 'ct.nii.gz'))
ct_image = pv.read(ct_path)

###############################################################################
# Load Segmentation Masks
# =======================
# Load all segmentation masks as a :class:`~pyvista.MultiBlock`.
#
# Get filepaths for all segmentation images.
seg_folder = Path(dataset_folder, 'segmentations')
seg_paths = sorted([str(Path(seg_folder, label)) for label in os.listdir(seg_folder)])

###############################################################################
# Extract the label names. The name of each file (without the extension)
# corresponds to the name of the label (e.g. ``'heart'`` or ``'skull'``).
label_names = [Path(path).name.replace('.nii.gz', '') for path in seg_paths]

###############################################################################
# Load images into a MultiBlock. Use a dictionary to map the label names
# to the images.
seg_dict = {name: pv.read(path) for name, path in zip(label_names, seg_paths)}
seg_multiblock = pv.MultiBlock(seg_dict)

###############################################################################
# Show Volumes
# ============
# Here, we plot the CT image along with a single segmentation mask of the
# heart.
#
# To visualize a particular segment as a volume, we first apply its mask
# to the CT image.
#
# Initialize an array with CT background values. Here, we set the image values
# to ``-1000`` which typically corresponds to air (low density) for calibrated
# CT images.
ct_array = ct_image.active_scalars
ct_background_value = -1000
ct_blank = np.ones_like(ct_array) * ct_background_value

###############################################################################
# .. note::
#    Alternatively the background CT value can be computed dynamically as the
#    mode:
#
#    .. code:: python
#       import statistics
#       ct_background_value = statistics.mode(ct_array)
#
#    Or may be retrieved from the image metadata (if the metadata is correct):
#
#    .. code:: python
#        nifti_header = pv.get_reader(ct_path).reader.GetNIFTIHeader()
#        ct_background_value = nifti_header.GetSclInter()

###############################################################################
# Extract the intensities for the ``'heart'`` segment. We use the blank
# CT array and update it to only include CT values for the heart region.
mask_array = seg_multiblock['heart'].active_scalars
heart_array = ct_blank.copy()
heart_array[mask_array == 1] = ct_array[mask_array == 1]

# Add the masked array to the CT image as a new set of scalar values.
ct_image['heart'] = heart_array

###############################################################################
# Create the plot.
#
# For the CT image, the opacity is set to a sigmoid function to show the
# subject's skeleton. Since different images have different intensity
# distributions, you may need to experiment with different sigmoid functions.
# See :func:`~pyvista.Plotter.add_volume` for details.

pl = pv.Plotter()

# Add the CT image.
pl.add_volume(
    ct_image,
    scalars='NIFTI',
    cmap='bone',
    opacity='sigmoid_15',
    show_scalar_bar=False,
)

# Add masked CT image of the heart and use a contrasting color map.
_ = pl.add_volume(
    ct_image,
    scalars='heart',
    cmap='gist_heat',
    opacity='linear',
    opacity_unit_distance=np.mean(ct_image.spacing),
)

# Orient the camera to provide a lateral-anterior view.
pl.view_yz()
pl.camera.azimuth = 70
pl.camera.up = (0, 0, 1)
pl.camera.zoom(1.5)
pl.show()

###############################################################################
# Generate Segment Surfaces
# -------------------------
# In this section, :func:`~pyvista.ImageDataFilters.contour_labeled` is used
# to generate surfaces from segmentation masks. Surfaces can be generated
# from each mask independently or all at once from a label map.

###############################################################################
# Independent Surfaces
# ====================
# Generate separate surfaces for the skull and the brain.
skull_mask = seg_multiblock['skull']
brain_mask = seg_multiblock['brain']
skull_surf = skull_mask.contour_labeled(smoothing=True)
brain_surf = brain_mask.contour_labeled(smoothing=True)

###############################################################################
# Plot the surfaces. The skull is clipped for visualization.

skull_surf_clipped = skull_surf.clip('z', origin=brain_surf.center)
pl = pv.Plotter()
_ = pl.add_mesh(skull_surf_clipped, color='cornsilk')
_ = pl.add_mesh(brain_surf, color='tomato')

# Orient the camera to provide an anterior view.
pl.view_zx()
pl.add_axes()
pl.camera.up = (0, 0, 1)
pl.show()

###############################################################################
# Note that the two meshes intersect and overlap in some areas. This can occur
# when smoothing is enabled and voxels from two masks share a face or edge.

###############################################################################
# Surfaces From Label Map
# =======================
# In the previous section, a call to :func:`~pyvista.ImageDataFilters.contour_labeled`
# was made to generate each surface separately. This is inefficient, since
# the underlying `vtkSurfaceNets3D <https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html>`_,
# filter can process multiple labels at once without requiring any additional
# iterations of the algorithm. This can greatly reduce processing times for
# datasets with many labels.
#
# To process all labels at once, we first create a label map, which is an
# alternative representation of the segmentation where the masks are
# combined into a single scalar array. These two formats are usually
# interchangeable, though a label map is more restrictive since it only allows
# a single voxel to map to one label.

# Generate integer ids for the labels. Here, we use the MultiBlock indices as
# the ids. The indices are offset by 1 since 0 is reserved for the background.
label_ids = [idx + 1 for idx in range(seg_multiblock.n_blocks)]

# Initialize array with background values (zeros)
label_map_array = np.zeros((seg_multiblock[0].n_points,), dtype=np.uint8)

# Add mask arrays
for i, name in zip(label_ids, label_names):
    label_map_array[seg_multiblock[name].active_scalars == 1] = i

###############################################################################
# .. note::
#    The label ids here are ordered alphabetically. This differs from the
#    ordering of the official label ids used by TotalSegmentator.

###############################################################################
# Create a new image to store the label map array.
label_map_image = seg_multiblock[0].copy()
label_map_image.clear_data()
label_map_image['label_map'] = label_map_array

###############################################################################
# Generate surfaces for all segments.
labels_mesh = label_map_image.contour_labeled(smoothing=True)

###############################################################################
# Plot the label map. Use a categorical colormap.
pl = pv.Plotter()
_ = pl.add_mesh(labels_mesh, cmap='glasbey', show_scalar_bar=False)

# Orient the camera to provide an anterior view.
pl.view_zx()
pl.camera.up = (0, 0, 1)
pl.camera.zoom(1.3)
pl.show()
