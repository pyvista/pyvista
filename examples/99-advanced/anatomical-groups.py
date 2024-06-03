"""
.. _anatomical_groups:

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

from __future__ import annotations

import pyvista as pv
from pyvista import examples

###############################################################################
# Load Dataset
# ------------
# Load a TotalSegmentator dataset. Can be the
# :func:`male <pyvista.examples.downloads.download_whole_body_ct_male>` or
# :func:`female <pyvista.examples.downloads.download_whole_body_ct_female>` female
# subjects included with `PyVista`, or any other subject from the
# `TotalSegmentator dataset <https://zenodo.org/records/10047292>`_.
dataset = examples.download_whole_body_ct_female()

###############################################################################
# Load the label map used for plotting different anatomical groups. See the
# <TODO ADD _medical_labels ref> for details on how to create a label map.
label_map = dataset['label_map']

###############################################################################
# Get metadata associated with the dataset

# Get the dict which maps the label names to the integer ids in the label map
label_ids_dict = dataset.user_dict['ids']
label_names = label_ids_dict.keys()
label_ids = label_ids_dict.values()

# Get the RGB color mapping
colors_dict = dataset.user_dict['colors']

###############################################################################
# Color Mapping
# =============
# Show the color mapping included with the dataset.
#
# .. raw:: html
#
#    <details><summary>Click show the full list of labels and colors.</summary>
#
#   .. note::
#       This mapping is included with the PyVista
#       :func:`male <pyvista.examples.downloads.download_whole_body_ct_male>` and
#       :func:`female <pyvista.examples.downloads.download_whole_body_ct_female>`
#       whole body datasets.
#
# Print the dictionary and format it to visually align the RGB values.
print(
    "{\n"
    + "\n".join(
        f'    {"'" + name + "':":<32} ({R:>3}, {G:>3}, {B:>3})'
        for name, (R, G, B) in colors_dict.items()
    )
    + "\n}",
)

# .. raw:: html
#
#    </details>


###############################################################################
# Utility Functions
# -----------------
# Define a few helper functions to visualize the data.
#
# To visual a particular group of anatomic regions, we first define a function
# to filter the labels by name. Given a list of terms, the function returns
# any label which contains any of the search terms.
def filter_labels(search_terms: list[str]):
    def include_label(label_name: str):
        return any(target in label_name for target in search_terms)

    return [label for label in label_names if include_label(label)]


###############################################################################
# We also define a look-up function to return the colors for a subset of the
# labels.
def lookup_colors(labels: list[str]):
    return [colors_dict[label] for label in labels]


###############################################################################
# Define a similar look-up for the label ids.
def lookup_ids(labels: list[str]):
    return [label_ids_dict[label] for label in labels]


###############################################################################
# Plotting Function
# -----------------
# Define a function which, given a list of terms, will lookup labels associated
# with those terms, generate contours for the labels, and plot the result.
def plot_anatomy(search_terms: list[str]):
    # Get a list of labels which contain any of the listed terms.
    group_names = filter_labels(search_terms)

    # Get the label ids corresponding to the matched labels.
    group_ids = lookup_ids(group_names)

    # Get the label colors to use for plotting.
    group_colors = lookup_colors(group_names)

    # Selectively generate surfaces for the specified labels.
    group_surface = dataset['label_map'].contour_labeled(select_inputs=group_ids, smoothing=True)

    # Split the labeled surface into separate meshes, one for each label
    split_surfaces = group_surface.extract_values(group_ids, split=True)

    # Plot the label map.
    # Allow empty meshes since some of the labels may not exist.
    pv.global_theme.allow_empty_mesh = True
    pl = pv.Plotter()
    [pl.add_mesh(surf, color) for surf, color in zip(split_surfaces, group_colors)]
    pl.view_zx()
    pl.camera.up = (0, 0, 1)
    pl.show()


###############################################################################
# Cardiovascular System
# ---------------------
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

###############################################################################
# Gastrointestinal System
# -----------------------
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


###############################################################################
# Spine
# -----
# Show segments of the spinal.

# Define terms which describe all relevant segments.
spine = [
    'sacrum',
    'vertebrae',
    'spinal_cord',
]

# Plot the labels associated with these terms.
plot_anatomy(spine)


###############################################################################
# Other Organs
# ------------
# Show other organs not included in the cardiovascular or gastrointestinal
# systems.

# Define terms which describe all relevant segments.
other_organs = [
    'spleen',
    'kidney',
    'brain',
    'gallbladder',
    'liver',
    'pancreas',
    'adrenal_gland',
    'lung',
    'trachea',
    'thyroid_gland',
    'prostate',
]

# Plot the labels associated with these terms.
plot_anatomy(other_organs)


###############################################################################
# Muscles
# -------
# Show the muscles.

# Define terms which describe all relevant segments.
muscles = [
    'gluteus',
    'autochthon',
    'iliopsoas',
]

# Plot the labels associated with these terms.
plot_anatomy(muscles)

###############################################################################
# Ribs
# ----
# Show the ribs.

# Define terms which describe all relevant segments.
ribs = [
    'rib',
    'sternum',
    'costal_cartilages',
]

# Plot the labels associated with these terms.
plot_anatomy(ribs)


###############################################################################
# Skeleton
# --------
# Show the skeleton.

# Define terms which describe all relevant segments.
skeleton = [
    'skull',
    'clavicula',
    'scapula',
    'humerus',
    'sacrum',
    'vertebrae',
    'rib',
    'sternum',
    'costal_cartilages',
    'hip',
    'femur',
]

# Plot the labels associated with these terms.
plot_anatomy(skeleton)
