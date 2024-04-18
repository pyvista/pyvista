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

import os
from pathlib import Path
import statistics

import numpy as np

import pyvista as pv
from pyvista import examples

###############################################################################
# Load Dataset
# ------------
# Load a medical image and its corresponding binary segmentation masks.
#
# For this example, we use data from the TotalSegmentator dataset which can be
# downloaded from `zenodo <https://zenodo.org/records/10047292>`_. Two subjects
# from this source, one male (``'s1397'``) and one female (``'s1380'``), are
# available for download in PyVista.

# Download the files associated with the male dataset.
dataset_folder = examples.download_whole_body_ct_male(load=False)

# The dataset contains a single CT image and a folder with 117 segmentation
# mask images, one for each of the 117 segmented anatomical structures.

os.listdir(dataset_folder)
len(os.listdir(Path(dataset_folder, 'segmentations')))

###############################################################################
# Load CT Image
# =============
# Read the image.
ct_path = str(Path(dataset_folder, 'ct.nii.gz'))
ct_image = pv.read(ct_path)

# Plot the image data as a volume. Set the opacity to a sigmoid function to
# show an outline of the subject. Since different images have different
# intensity distributions, you may need to experiment with different sigmoid
# functions. See :func:`:func:`~pyvista.Plotter.add_volume` for details.
pl = pv.Plotter()
_ = pl.add_volume(
    ct_image,
    cmap='bone',
    opacity='sigmoid_9',
)
pl.view_zx()
pl.camera.up = (0, 0, 1)
pl.show()

###############################################################################
# Load Segmentation Masks
# =======================
# Load all segmentation masks as a :class:`~pyvista.MultiBlock`.
#
# Get filepaths for all segmentation images.
seg_folder = Path(dataset_folder, 'segmentations')
seg_paths = sorted([str(Path(seg_folder, label)) for label in os.listdir(seg_folder)])

# Extract the label names. The name of each file (without the extension)
# corresponds to the name of the label (e.g. ``'heart'`` or ``'skull'``).
# See :ref:`label_names` for a complete list of labels.
label_names = [Path(path).name.replace('.nii.gz', '') for path in seg_paths]

# Create a :class:`~pyvista.MultiBlock` from a dictionary which maps each
# label to its respective :class:`~pyvista.ImageData` object.
seg_dict = {name: pv.read(path) for name, path in zip(label_names, seg_paths)}
seg_multiblock = pv.MultiBlock(seg_dict)

# To visualize the CT image intensities for a particular segment, we can
# apply the mask to the CT image and plot it as a separate volume.

# First, create a 'blank' CT image. For this example, we assume most of
# the image is background and compute the background value as the mode. For
# calibrated CT images, a value of ``-1000`` corresponds to air.
ct_array = ct_image.active_scalars
ct_background_value = statistics.mode(ct_array)
ct_background_value
ct_blank = np.ones_like(ct_array) * ct_background_value

# Extract the intensities for the ``'heart'`` segment. We use the blank
# CT array and update it to only include CT values for the heart region.
mask_array = seg_multiblock['heart'].active_scalars
heart_array = ct_blank.copy()
heart_array[mask_array == 1] = ct_array[mask_array == 1]
ct_image['heart'] = heart_array

# Plot the intensities of the masked heart volume.
pl = pv.Plotter()
_ = pl.add_volume(
    ct_image,
    scalars='heart',
    cmap='gist_heat',
    opacity='linear',
    opacity_unit_distance=np.mean(ct_image.spacing),
)
# Add the full CT image to the plot for context.
_ = pl.add_volume(
    ct_image,
    cmap='bone',
    opacity='sigmoid_15',
    show_scalar_bar=False,
)
# Orient the camera to provide a lateral-anterior view.
pl.view_yz()
pl.camera.azimuth = 60
pl.camera.up = (0, 0, 1)
pl.camera.zoom(2)
pl.show()

###############################################################################
# Generate Segment Surfaces
# -------------------------
# In this section, :func:`~pyvista.ImageDataFilters.contour_labeled` is mainly
# used to generate surfaces from segmentation masks. Surfaces can be generated
# from each mask independently or all at once from a label map.
#
# Generate separate surfaces for the skull and the brain.
skull_mask = seg_multiblock['skull']
brain_mask = seg_multiblock['brain']
skull_surf = skull_mask.contour_labeled(smoothing=True)
brain_surf = brain_mask.contour_labeled(smoothing=True)

# Plot the surfaces. The skull is clipped for visualization.
skull_surf_clipped = skull_surf.clip('z', origin=brain_surf.center)
pl = pv.Plotter()
_ = pl.add_mesh(skull_surf_clipped, color='cornsilk')
_ = pl.add_mesh(brain_surf, color='tomato')

# Orient the camera to provide a lateral-anterior view.
pl.view_yz()
pl.camera.azimuth = 45
pl.camera.up = (0, 0, 1)
pl.camera.zoom(1.5)
pl.show()

# Note that the two meshes intersect and overlap in some areas. This can occur
# when smoothing is enabled and voxels from two masks share a face or edge.

###############################################################################
# Create Label Map
# ================
# In the previous section, a call to :func:`~pyvista.ImageDataFilters.contour_labeled`
# was made to generate each surface separately. This is inefficient, since
# the underlying `vtkSurfaceNets3D <https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html>`_,
# filter can process multiple labels at once without requiring any additional
# iterations of the algorithm. This can greatly reduce processing times for
# datasets with many labels, such as the whole body datasets
# :func:`~pyvista.examples.downloads.download_whole_body_ct_female` and
# :func:`~pyvista.examples.downloads.download_whole_body_ct_male` which have
# 117.
#
# To process all labels at once, we first create a label map. A label map
# is an alternative representation of the segmentation where the masks are
# combined into a single scalar array. These two formats are often
# interchangeable, except in cases where the segmentation masks overlap, since
# a label map maps each voxel to exactly one label.
#
# Generate integer ids for the labels. Here, we use the MultiBlock indices as
# the ids. The indices are offset by 1 since 0 is reserved for the background.
label_ids = [idx + 1 for idx in range(seg_multiblock.n_blocks)]
# Initialize array with background values (zeros).
label_map_array = np.zeros((seg_multiblock[0].n_points,), dtype=np.uint8)
for i, name in zip(label_ids, label_names):
    # Add mask arrays
    label_map_array[seg_multiblock[name].active_scalars == 1] = i

# Create a new image to store the label map array.
label_map_image = seg_multiblock[0].copy()
label_map_image.clear_data()
label_map_image['label_map'] = label_map_array

# Use the label map to generate surface meshes for all segments.
labels_mesh = label_map_image.contour_labeled(smoothing=True)

# Plot the label map.
pl = pv.Plotter()
_ = pl.add_mesh(labels_mesh, cmap='glasbey', show_scalar_bar=False)
pl.view_zx()
pl.camera.up = (0, 0, 1)
pl.camera.zoom(1.3)
pl.show()
