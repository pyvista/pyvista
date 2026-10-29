"""
.. _image_library_accessors_example:

Extend Images with Pillow and SciPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add Pillow and SciPy filters to :class:`~pyvista.ImageData` as dataset accessors.

The accessors are registered with :func:`~pyvista.register_dataset_accessor`, and
:attr:`~pyvista.DataSetAttributes.reshaped` passes arrays between the libraries.

Each library expects its own array layout. Pillow uses ``(row, column,
channel)`` with the top row first, read and written by the ``'image'``
preset. :mod:`scipy.ndimage` takes volumes indexed along x, y and z, read and
written by the default ``'grid'`` preset. With the reshaping handled, each
accessor method is a few lines, and its output chains with PyVista filters.
See :ref:`reshape_data_arrays_example` for every preset and
:ref:`extending-pyvista` for accessors.

"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFilter
import pyvista as pv
from pyvista import examples
from scipy import ndimage

# %%
# A Pillow accessor
# +++++++++++++++++
# ``to_pil`` reads an array in Pillow's layout and ``apply`` writes the result
# of any Pillow function back to an image with the same origin and spacing.


@pv.register_dataset_accessor('pil', pv.ImageData)
class PillowAccessor:
    """Pillow image operations for 2D images."""

    def __init__(self, image):
        self._image = image

    def to_pil(self, scalars=None):
        """Return an array of the image as a Pillow image."""
        name = scalars or self._image.active_scalars_name
        pixels = self._image.point_data.reshaped.using('image')[name]
        return Image.fromarray(np.ascontiguousarray(pixels))

    def apply(self, function, *, scalars=None):
        """Return a new image from a function of a Pillow image."""
        name = scalars or self._image.active_scalars_name
        result = function(self.to_pil(name))
        output = pv.ImageData(
            dimensions=(*result.size, 1),
            spacing=self._image.spacing,
            origin=self._image.origin,
        )
        output.point_data.reshaped.using('image')[name] = np.asarray(result)
        return output

    def filter(self, image_filter, *, scalars=None):
        """Return a new image filtered by a Pillow image filter."""
        return self.apply(lambda image: image.filter(image_filter), scalars=scalars)


# %%
# Read the PyVista logo and blur it with a Pillow filter.
logo = pv.read(examples.logofile)
blurred = logo.pil.filter(ImageFilter.GaussianBlur(8))
blurred.plot(rgb=True, cpos='xy', background='white')

# %%
# The Pillow image and the PyVista image show the same picture, upright.
fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(blurred.pil.to_pil())
ax.set_axis_off()
plt.show()

# %%
# The output is an ordinary :class:`~pyvista.ImageData`, so Pillow and PyVista
# filters chain. Blur the opacity of the logo and warp it into a relief.
height = logo.pil.apply(
    lambda image: image.getchannel('A').filter(ImageFilter.GaussianBlur(6))
)
height.point_data['PNGImage'] = height.point_data['PNGImage'] / 255
relief = height.warp_by_scalar('PNGImage', factor=60)
pl = pv.Plotter()
pl.add_mesh(
    relief, scalars=logo.point_data['PNGImage'][:, :3], rgb=True, smooth_shading=True
)
pl.view_xy()
pl.camera.elevation = -50
pl.show()

# %%
# A SciPy accessor
# ++++++++++++++++
# :mod:`scipy.ndimage` filters work on volumes of any dimension, indexed like
# the default ``'grid'`` preset. Convert the filter sizes from physical units
# to voxels with the image spacing.


@pv.register_dataset_accessor('ndimage', pv.ImageData)
class NdimageAccessor:
    """SciPy n-dimensional image operations."""

    def __init__(self, image):
        self._image = image

    def _volume(self, scalars):
        """Return the named point array in the grid shape."""
        return self._image.point_data.reshaped[scalars or self._image.active_scalars_name]

    def gaussian_filter(self, sigma, *, scalars=None):
        """Return a copy smoothed by a Gaussian with ``sigma`` in physical units."""
        output = self._image.copy()
        name = scalars or self._image.active_scalars_name
        voxels = np.divide(sigma, self._image.spacing)
        output.point_data.reshaped[name] = ndimage.gaussian_filter(
            self._volume(name), voxels
        )
        return output

    def label(self, threshold, *, scalars=None):
        """Return a copy with connected regions above ``threshold`` labeled."""
        output = self._image.copy()
        labels, _ = ndimage.label(self._volume(scalars) > threshold)
        output.point_data.reshaped['labels'] = labels
        return output


# %%
# Build a noisy volume of a few blobs, smooth it, and label the blobs.
rng = np.random.default_rng(0)
dimensions = (80, 60, 40)
volume = pv.ImageData(dimensions=dimensions, spacing=(0.5, 0.5, 0.5))
centers = rng.uniform((5, 5, 5), (35, 25, 15), size=(6, 3))
distance = np.min(
    [np.linalg.norm(volume.points - center, axis=1) for center in centers], axis=0
)
volume.point_data['density'] = np.exp(-(distance**2) / 8) + rng.normal(
    0, 0.15, volume.n_points
)

smooth = volume.ndimage.gaussian_filter(1.0)
labeled = smooth.ndimage.label(0.5)
np.unique(labeled.point_data['labels'])

# %%
# Compare a slice of the noisy and the smoothed volume. The reshaped arrays
# plot directly with Matplotlib, with x along the horizontal axis.
k = dimensions[2] // 2
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for ax, image, title in zip(axes, [volume, smooth], ['noisy', 'smoothed'], strict=True):
    ax.imshow(
        image.point_data.reshaped['density'][:, :, k].T, origin='lower', cmap='magma'
    )
    ax.set_title(title)
    ax.set_axis_off()
plt.show()

# %%
# Show the labeled blobs in 3D.
blobs = labeled.threshold(0.5, scalars='labels', all_scalars=True)
blobs.plot(scalars='labels', cmap='tab10', show_scalar_bar=False)

# %%
# Remove the accessors when they are no longer needed.
pv.unregister_dataset_accessor('pil', pv.ImageData)
pv.unregister_dataset_accessor('ndimage', pv.ImageData)
# %%
# .. tags:: filter
