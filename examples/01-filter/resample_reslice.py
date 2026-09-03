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

"""

from __future__ import annotations

import numpy as np

# sphinx_gallery_thumbnail_number = 3
import pyvista as pv
from pyvista import examples

# %%
# Resample
# ++++++++
#
# Load a photograph and down-sample it to an eighth of its resolution. Anti-aliasing
# blurs the image before sampling it, which reduces the artifacts caused by discarding
# samples.

gourds = examples.download_gourds()
downsampled = gourds.resample(1 / 8, 'lanczos')
smoothed = gourds.resample(1 / 8, 'lanczos', anti_aliasing=True)

pl = pv.Plotter(shape=(1, 3))
pl.add_mesh(gourds, rgba=True, lighting=False)
pl.add_text('Original', font_size=10)
pl.subplot(0, 1)
pl.add_mesh(downsampled, rgba=True, lighting=False)
pl.add_text('Down-sampled', font_size=10)
pl.subplot(0, 2)
pl.add_mesh(smoothed, rgba=True, lighting=False)
pl.add_text('Anti-aliased', font_size=10)
pl.link_views()
pl.view_xy()
pl.camera.tight()
pl.show()

# %%
# The image has fewer samples, but it still covers the same region of space.

print(gourds.dimensions, downsampled.dimensions)
print(gourds.bounds)
print(downsampled.bounds)

# %%
# Reslice
# +++++++
#
# Load a second photograph. It is smaller than the first, and both images start at the
# origin with unit spacing, so the bird covers the lower left corner of the region the
# gourds cover.

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
# Oblique Slices
# ++++++++++++++
#
# Because the reference defines the sampling points, it may also be rotated. Build a
# grid through the centre of a brain volume, tilted 30 degrees.

brain = examples.download_brain()

rotation = pv.Transform().rotate_vector((1, 0, 0), 30).matrix[:3, :3]
dimensions = np.array([200, 200, 1])
spacing = np.array([1.0, 1.0, 1.0])

oblique = pv.ImageData(dimensions=dimensions, spacing=spacing)
oblique.direction_matrix = rotation
oblique.origin = np.array(brain.center) - rotation @ (spacing * (dimensions - 1) / 2)

# %%
# Show the plane cutting through the volume.

pl = pv.Plotter()
pl.add_volume(brain, cmap='bone', opacity='sigmoid_5', show_scalar_bar=False)
pl.add_mesh(oblique.points_to_cells(), color='red', style='wireframe', opacity=0.5)
pl.view_vector((1, -1, 0.4), viewup=(0, 0, 1))
pl.show()

# %%
# Reslice the volume onto that grid to sample the oblique plane.

sliced = brain.reslice(oblique, 'linear')

# %%
# The result is an image in the plane of the reference. Reset its orientation to view
# it face-on.

flat = sliced.copy()
flat.direction_matrix = np.eye(3)
flat.origin = (0.0, 0.0, 0.0)

pl = pv.Plotter()
pl.add_mesh(flat, cmap='bone', show_scalar_bar=False, lighting=False)
pl.view_xy()
pl.camera.tight()
pl.show()

# %%
# .. tags:: filter
