# ruff: flake8: noqa: E241, E201
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

import pyvista as pv
from pyvista import examples

###############################################################################
# Color Mapping
# -------------
# Define the color mapping for all labels.
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

# fmt: off
colors = {
    'adrenal_gland_left':           (249, 186, 150),
    'adrenal_gland_right':          (249, 186, 150),
    'aorta':                        (224,  97,  76),
    'atrial_appendage_left':        (  0, 161, 196),
    'autochthon_left':              (171,  85,  68),
    'autochthon_right':             (171,  85,  68),
    'brachiocephalic_trunk':        (196, 121,  79),
    'brachiocephalic_vein_left':    (  0, 161, 196),
    'brachiocephalic_vein_right':   (  0, 151, 206),
    'brain':                        (250, 250, 225),
    'clavicula_left':               (205, 179, 108),
    'clavicula_right':              (205, 179, 108),
    'colon':                        (204, 168, 143),
    'common_carotid_artery_left':   (  0, 112, 165),
    'common_carotid_artery_right':  ( 10, 105, 155),
    'costal_cartilages':            (111, 184, 210),
    'duodenum':                     (255, 253, 229),
    'esophagus':                    (211, 171, 143),
    'femur_left':                   (255, 238, 170),
    'femur_right':                  (255, 238, 170),
    'gallbladder':                  (139, 150,  98),
    'gluteus_maximus_left':         (188,  95,  76),
    'gluteus_maximus_right':        (188,  95,  76),
    'gluteus_medius_left':          (178, 105,  76),
    'gluteus_medius_right':         (178, 105,  76),
    'gluteus_minimus_left':         (178,  95,  86),
    'gluteus_minimus_right':        (178,  95,  86),
    'heart':                        (206, 110,  84),
    'hip_left':                     (212, 188, 102),
    'hip_right':                    (212, 188, 102),
    'humerus_left':                 (205, 179, 108),
    'humerus_right':                (205, 179, 108),
    'iliac_artery_left':            (216, 101,  79),
    'iliac_artery_right':           (216, 101,  79),
    'iliac_vena_left':              (  0, 151, 206),
    'iliac_vena_right':             (  0, 151, 206),
    'iliopsoas_left':               (188,  95,  76),
    'iliopsoas_right':              (188,  95,  76),
    'inferior_vena_cava':           (  0, 151, 206),
    'kidney_cyst_left':             (205, 205, 100),
    'kidney_cyst_right':            (205, 205, 100),
    'kidney_left':                  (185, 102,  83),
    'kidney_right':                 (185, 102,  83),
    'liver':                        (221, 130, 101),
    'lung_lower_lobe_left':         (224, 186, 162),
    'lung_lower_lobe_right':        (224, 186, 162),
    'lung_middle_lobe_right':       (202, 164, 140),
    'lung_upper_lobe_left':         (172, 138, 115),
    'lung_upper_lobe_right':        (172, 138, 115),
    'pancreas':                     (249, 180, 111),
    'portal_vein_and_splenic_vein': (  0, 151, 206),
    'prostate':                     (230, 158, 140),
    'pulmonary_vein':               (  0, 122, 171),
    'rib_left_1':                   (253, 232, 158),
    'rib_left_10':                  (253, 232, 158),
    'rib_left_11':                  (253, 232, 158),
    'rib_left_12':                  (253, 232, 158),
    'rib_left_2':                   (253, 232, 158),
    'rib_left_3':                   (253, 232, 158),
    'rib_left_4':                   (253, 232, 158),
    'rib_left_5':                   (253, 232, 158),
    'rib_left_6':                   (253, 232, 158),
    'rib_left_7':                   (253, 232, 158),
    'rib_left_8':                   (253, 232, 158),
    'rib_left_9':                   (253, 232, 158),
    'rib_right_1':                  (253, 232, 158),
    'rib_right_10':                 (253, 232, 158),
    'rib_right_11':                 (253, 232, 158),
    'rib_right_12':                 (253, 232, 158),
    'rib_right_2':                  (253, 232, 158),
    'rib_right_3':                  (253, 232, 158),
    'rib_right_4':                  (253, 232, 158),
    'rib_right_5':                  (253, 232, 158),
    'rib_right_6':                  (253, 232, 158),
    'rib_right_7':                  (253, 232, 158),
    'rib_right_8':                  (253, 232, 158),
    'rib_right_9':                  (253, 232, 158),
    'sacrum':                       (212, 188, 102),
    'scapula_left':                 (212, 188, 102),
    'scapula_right':                (212, 188, 102),
    'skull':                        (241, 213, 144),
    'small_bowel':                  (205, 167, 142),
    'spinal_cord':                  (244, 214,  49),
    'spleen':                       (157, 108, 162),
    'sternum':                      (244, 217, 154),
    'stomach':                      (216, 132, 105),
    'subclavian_artery_left':       (216, 101,  69),
    'subclavian_artery_right':      (216, 101,  89),
    'superior_vena_cava':           (  0, 141, 226),
    'thyroid_gland':                (220, 160,  30),
    'trachea':                      (182, 228, 255),
    'urinary_bladder':              (222, 154, 132),
    'vertebrae_C1':                 (255, 255, 207),
    'vertebrae_C2':                 (255, 255, 207),
    'vertebrae_C3':                 (255, 255, 207),
    'vertebrae_C4':                 (255, 255, 207),
    'vertebrae_C5':                 (255, 255, 207),
    'vertebrae_C6':                 (255, 255, 207),
    'vertebrae_C7':                 (255, 255, 207),
    'vertebrae_L1':                 (212, 188, 102),
    'vertebrae_L2':                 (212, 188, 102),
    'vertebrae_L3':                 (212, 188, 102),
    'vertebrae_L4':                 (212, 188, 102),
    'vertebrae_L5':                 (212, 188, 102),
    'vertebrae_S1':                 (212, 208, 122),
    'vertebrae_T1':                 (226, 202, 134),
    'vertebrae_T10':                (226, 202, 134),
    'vertebrae_T11':                (226, 202, 134),
    'vertebrae_T12':                (226, 202, 134),
    'vertebrae_T2':                 (226, 202, 134),
    'vertebrae_T3':                 (226, 202, 134),
    'vertebrae_T4':                 (226, 202, 134),
    'vertebrae_T5':                 (226, 202, 134),
    'vertebrae_T6':                 (226, 202, 134),
    'vertebrae_T7':                 (226, 202, 134),
    'vertebrae_T8':                 (226, 202, 134),
    'vertebrae_T9':                 (226, 202, 134),
}
# fmt: on

# .. raw:: html
#
#    </details>


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
