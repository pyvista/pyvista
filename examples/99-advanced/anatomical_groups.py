"""
.. _anatomical_groups_example:

Visualize Anatomical Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example visualizes different anatomical groups using the segmentation
labels available from the downloadable datasets
:func:`~pyvista.examples.downloads.download_whole_body_ct_female` and
:func:`~pyvista.examples.downloads.download_whole_body_ct_male`.

These datasets include labels for 117 anatomical structures. In this example,
the labels are grouped by filtering the list of labels and coloring the
labels with the recommended RGB values used by the 3DSlicer
`TotalSegmentator Extension <https://github.com/lassoan/SlicerTotalSegmentator>`_.
"""

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
# Load Dataset
# ------------
# Load a TotalSegmentator dataset. Can be the
# :func:`male <pyvista.examples.downloads.download_whole_body_ct_male>` or
# :func:`female <pyvista.examples.downloads.download_whole_body_ct_female>` female
# subjects included with `PyVista`, or any other subject from the
# `TotalSegmentator dataset <https://zenodo.org/records/10047292>`_.
dataset = examples.download_whole_body_ct_female()

# %%
# Get the label map used for plotting different anatomical groups.
label_map = dataset['label_map']

# %%
# Get metadata associated with the dataset

# Get a list of all label names. This list will be filtered by group.
label_names = dataset['segmentations'].keys()

# Get color and id mappings included with the dataset. These are used to filter and
# color the contours.
names_to_colors = dataset.user_dict['names_to_colors']
names_to_ids = dataset.user_dict['names_to_ids']
ids_to_colors = dataset.user_dict['ids_to_colors']

# %%
# Color Mapping
# =============
# Show the color mapping included with the dataset. Print the dictionary and format
# it to visually align the RGB values. The formatted dictionary is valid python code.
#
# .. note
#
#     This mapping is specific to the PyVista datasets and is not part of the
#     TotalSegmentator data.

print('{')
for name, (R, G, B) in names_to_colors.items():
    print(f'{name!r:<32}: ({R:>3}, {G:>3}, {B:>3}),')
print('}')

# %%
# Utility Functions
# -----------------
# Define helper functions to visualize the data.


# %%
# filter_labels
# =============
# To visualize a particular group of anatomic regions, we first define a function
# to filter the labels by name. Given a list of terms, the function returns
# any label which contains any of the search terms.
def filter_labels(label_names: list[str], search_terms: list[str]):
    def include_label(label_name: str):
        return any(target in label_name for target in search_terms)

    return [label for label in label_names if include_label(label)]


# %%
# plot_anatomy
# ============
# Define a function which, given a list of terms, will look up labels associated
# with those terms, generate contours for the labels, and plot the result. The function
# uses :meth:`~pyvista.ImageDataFilters.contour_labels` for generating contours and
# :meth:`~pyvista.DataSetFilters.color_labels` for coloring them.
def plot_anatomy(search_terms: list[str]):
    # Get a list of labels which contain any of the listed terms.
    group_names = filter_labels(label_names, search_terms)

    # Get the label ids corresponding to the matched labels.
    group_ids = [names_to_ids[name] for name in group_names]

    # Selectively generate surfaces for the specified labels
    group_surface = dataset['label_map'].contour_labels(select_inputs=group_ids)

    # Color the labels with the color mapping
    colored_surface = group_surface.color_labels(colors=ids_to_colors)

    # Plot the label map.
    pl = pv.Plotter()
    pl.add_mesh(colored_surface)
    pl.view_zx()
    pl.camera.up = (0, 0, 1)
    pl.show()


# %%
# Anatomical Groups
# -----------------
# Group the labels and visualize the result.

# %%
# Cardiovascular System
# =====================
# Show segments of the cardiovascular system.

# Define terms which describe all relevant segments.
cardio = [
    'heart',
    'aorta',
    'artery',
    'brachiocephalic_trunk',
    'vein',
    'atrial_appendage',
    'vena_cava',
]

# Plot the labels associated with these terms.
plot_anatomy(cardio)

# %%
# Gastrointestinal System
# =======================
# Show segments of the gastrointestinal system.

# Define terms which describe all relevant segments.
gastro = [
    'esophagus',
    'stomach',
    'duodenum',
    'small_bowel',
    'colon',
    'urinary_bladder',
]

# Plot the labels associated with these terms.
plot_anatomy(gastro)

# %%
# Spine
# =====
# Show segments of the spinal.

# Define terms which describe all relevant segments.
spine = [
    'spinal_cord',
    'vertebrae',
    'sacrum',
]

# Plot the labels associated with these terms.
plot_anatomy(spine)

# %%
# Other Organs
# ============
# Show other organs not included in the cardiovascular or gastrointestinal
# systems.

# Define terms which describe all relevant segments.
other_organs = [
    'brain',
    'spinal_cord',
    'thyroid_gland',
    'trachea',
    'lung',
    'adrenal_gland',
    'spleen',
    'liver',
    'gallbladder',
    'kidney',
    'pancreas',
    'prostate',
]

# Plot the labels associated with these terms.
plot_anatomy(other_organs)

# %%
# Muscles
# =======
# Show the muscles.

# Define terms which describe all relevant segments.
muscles = [
    'gluteus',
    'autochthon',
    'iliopsoas',
]

# Plot the labels associated with these terms.
plot_anatomy(muscles)

# %%
# Ribs
# ====
# Show the ribs.

# Define terms which describe all relevant segments.
ribs = [
    'rib',
    'sternum',
    'costal_cartilages',
]

# Plot the labels associated with these terms.
plot_anatomy(ribs)

# %%
# Skeleton
# ========
# Show the skeleton.

# Define terms which describe all relevant segments.
skeleton = [
    'skull',
    'clavicula',
    'scapula',
    'humerus',
    'vertebrae',
    'sternum',
    'rib',
    'costal_cartilages',
    'hip',
    'sacrum',
    'femur',
]

# Plot the labels associated with these terms.
plot_anatomy(skeleton)

# %%
# .. tags:: medical
