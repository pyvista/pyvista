"""
.. _compare_datasets_example:

Comparing Datasets
~~~~~~~~~~~~~~~~~~

Compare the shoulder blades of two patients with :func:`~pyvista.plot_compare`.

Comparing datasets means drawing each one in its own subplot with the same
display settings, which :func:`~pyvista.plot_compare` does in a single call.
Here the left and right scapulae of two whole body CT scans are compared, which
takes more than plotting them side-by-side: the four bones are scanned in
different places, at different sizes, and facing different directions, and each
of those has to be dealt with before the shapes can be compared at all.
"""

# sphinx_gallery_thumbnail_number = 4
import pyvista as pv
from pyvista import examples

# %%
# Four Shoulder Blades
# ====================
# Load the :func:`male <pyvista.examples.downloads.download_whole_body_ct_male>`
# and :func:`female <pyvista.examples.downloads.download_whole_body_ct_female>`
# whole body CT scans and contour the scapula segmentations with
# :meth:`~pyvista.ImageDataFilters.contour_labels`.

loaders = {
    'male': examples.download_whole_body_ct_male,
    'female': examples.download_whole_body_ct_female,
}

scapulae = {}
for sex, loader in loaders.items():
    segmentations = loader()['segmentations']
    for side in ['left', 'right']:
        scapulae[f'{sex} {side}'] = segmentations[f'scapula_{side}'].contour_labels()

dataset_kwargs = {'color': 'ivory', 'show_scalar_bar': False}

# %%
# As Scanned
# ==========
# The bones are still in the coordinates of the scan they came from, metres
# apart from each other. Pass ``link=False`` so that each subplot gets its own
# camera, fit to its own bone. The keys of the dict are used as the labels.

pv.plot_compare(scapulae, link=False, **dataset_kwargs)

# %%
# Each bone is now visible, but nothing about the four can be compared. Every
# subplot is framed independently, so a bone which fills its subplot is not
# necessarily any larger than one which does not.

for name, scapula in scapulae.items():
    x, y, z = scapula.center
    print(f'{name:14} is centered at ({x:6.1f}, {y:6.1f}, {z:6.1f})')

# %%
# In a Common Frame
# =================
# :meth:`~pyvista.DataSetFilters.align_xyz` rotates each bone so that its
# longest axis lies along the x-axis, and centers it on the origin. The
# subplots share a single camera by default, and that camera is fit to all four
# bones at once, so they are now drawn to the same scale and can be compared.

aligned = {name: scapula.align_xyz() for name, scapula in scapulae.items()}

# pv.plot_compare(aligned, **dataset_kwargs)

# %%
# The male scapulae are visibly larger, which the bounds confirm.

for name, scapula in aligned.items():
    length, width, depth = scapula.bounds_size
    print(f'{name:14} is {length:5.1f} long, {width:5.1f} wide, {depth:4.1f} deep')

# %%
# A Consistent Orientation
# ========================
# The bones are drawn to the same scale, but one of them is upside down. The
# principal axes used for the alignment have arbitrary signs, so bones of the
# same shape can still end up rotated differently.
#
# Seeding the axes fixes the signs. The scapula's longest axis runs down the
# patient, and the bone faces out to the side it belongs to, which is the
# direction that differs between the left and the right.

aligned = {
    name: scapula.align_xyz(
        axis_0_direction='z',
        axis_1_direction='x' if 'left' in name else '-x',
    )
    for name, scapula in scapulae.items()
}

pv.plot_compare(aligned, **dataset_kwargs)

# %%
# Each side now agrees between the patients. The left and right bones remain
# mirror images of each other, as they are in the body, so it is the two
# columns which are worth comparing rather than the rows.

# %%
# A Common Yardstick
# ==================
# ``reference_mesh`` draws the same mesh in every subplot, styled separately
# from the bones with ``reference_kwargs``. Giving all four the bounding box of
# the largest bone turns the comparison into a measurement: the male scapulae
# fill the box and the female scapulae fall well short of it.

box = pv.Box(aligned['male left'].bounds)

# pv.plot_compare(
#     aligned,
#     reference_mesh=box,
#     reference_kwargs={'style': 'wireframe', 'color': 'red', 'line_width': 2},
#     **dataset_kwargs,
# )

# %%
# .. tags:: medical, plot
