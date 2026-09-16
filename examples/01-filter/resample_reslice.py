"""
.. _resample_reslice_example:

Resample and Reslice Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare the ``resample`` and ``reslice`` image filters.

:meth:`~pyvista.ImageDataFilters.resample` and :meth:`~pyvista.ImageDataFilters.reslice`
both create a new image by interpolating an existing one, but they answer different
questions. ``resample`` changes how densely an image is sampled and leaves it where it
is. ``reslice`` samples the image at the points of a second image, so the two images end
up on a common grid.

See :meth:`~pyvista.ImageDataFilters.resample` for changing an image's resolution and
for the interpolation and anti-aliasing options both filters share.

"""

from __future__ import annotations

import numpy as np

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

# %%
# Reslice
# +++++++
#
# Load two photographs. The bird is smaller than the gourds, and both start at the
# origin with unit spacing, so the bird covers the lower left corner of the region the
# gourds cover.

gourds = examples.download_gourds()
bird = examples.download_bird()
print(bird.dimensions, gourds.dimensions)

# %%
# ``reslice`` samples the bird at the points of the gourds image. The bird keeps its
# size and position, and the reference points which fall outside it take
# ``background_value``.

resliced = bird.reslice(gourds, 'linear', background_value=0)

# %%
# Plot the result with the outline of the gourds image in red. The bird fills the
# corner of the grid it occupies and the rest is background.

pl = pv.Plotter()
pl.add_mesh(resliced, rgba=True, lighting=False)
pl.add_mesh(gourds.outline(), color='red', line_width=3)
pl.view_xy()
pl.camera.tight()
pl.show()

# %%
# Position Matters
# ++++++++++++++++
#
# The difference is clearest when the reference covers only part of the image. Build a
# reference which covers the leftmost gourd at half the spacing.

reference = pv.ImageData(
    dimensions=(300, 300, 1), spacing=(0.5, 0.5, 1.0), origin=(60.0, 130.0, 0.0)
)

# %%
# ``reslice`` returns that region of the image, sampled at the reference's points.
# ``resample`` returns the whole image squeezed into the reference's dimensions.

resliced = gourds.reslice(reference, 'linear')
resampled = gourds.resample(dimensions=reference.dimensions, interpolation='linear')

pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(resliced, rgba=True, lighting=False)
pl.add_text('reslice', font_size=10)
pl.view_xy()
pl.camera.tight()
pl.subplot(0, 1)
pl.add_mesh(resampled, rgba=True, lighting=False)
pl.add_text('resample', font_size=10)
pl.view_xy()
pl.camera.tight()
pl.show()

# %%
# Both outputs have the same number of samples. The resliced image reports the
# reference's geometry, because that is where its samples were taken. The resampled
# image keeps the bounds of the gourds and only changes its spacing.

print(resliced.origin, resliced.spacing)
print(resampled.origin, resampled.spacing)

# %%
# Oblique Anatomy
# +++++++++++++++
#
# A structure which does not lie along the scan axes is hard to read in the slices the
# scanner produced. Load a whole-body CT with its segmentations and take the left
# scapula, a flat bone which sits at an angle to all three axes.

dataset = examples.download_whole_body_ct_male()
ct = dataset['ct']
scapula = dataset['segmentations']['scapula_left']

# %%
# Fit axes to the bone itself. The foreground of its mask gives the plane the blade
# lies in: the first two principal axes span that plane and the third is its normal.

foreground = scapula.points[scapula.active_scalars > 0]
center = foreground.mean(axis=0)
axes = pv.principal_axes(foreground)
extent = np.abs((foreground - center) @ axes.T).max(axis=0)

# %%
# Build one grid big enough to hold the blade, and a transform which brings the bone to
# the middle of it under a given rotation.

dimensions = (2 * extent[:2] + 20).astype(int)
reference = pv.ImageData(dimensions=(*dimensions, 1), spacing=(1.0, 1.0, 1.0))


def centered(rotation):
    """Return the matrix which centres the scapula in the reference under ``rotation``."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.array(reference.center) - rotation @ center
    return matrix


# %%
# Reslice twice from that one grid. Without a rotation the samples follow the scan axes;
# with the bone's own axes they follow the blade. Since the grid itself is never rotated,
# both outputs are ordinary axis-aligned images.

along_scan = ct.reslice(
    reference, 'linear', transform=centered(np.eye(3)), background_value=-1000
)
along_bone = ct.reslice(
    reference, 'linear', transform=centered(axes), background_value=-1000
)

# %%
# The blade crosses the scan plane at an angle, so it appears there as a thin sliver.
# Sampled along its own axes it is a single image, with the glenoid and the head of the
# humerus beside it.

pl = pv.Plotter(shape=(1, 2))
for index, (image, label) in enumerate(
    [(along_scan, 'scan axes'), (along_bone, 'bone axes')]
):
    pl.subplot(0, index)
    pl.add_mesh(
        image, cmap='bone', clim=[-200, 900], show_scalar_bar=False, lighting=False
    )
    pl.add_text(label, font_size=10)
    pl.view_xy()
    pl.camera.tight()
pl.show()

# %%
# .. tags:: filter
