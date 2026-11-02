"""
.. _image_library_accessors_example:

Extend Images With Pillow and SciPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add Pillow and SciPy filters to :class:`~pyvista.ImageData` as dataset accessors.

The accessors are registered with :func:`~pyvista.register_dataset_accessor`, and
:attr:`~pyvista.DataSetAttributes.reshaped` passes arrays between the libraries.

Each library expects its own array layout. Pillow uses ``(row, column,
channel)`` with the top row first, the layout of the ``'image'`` preset.
The SciPy accessor uses the default ``'grid'`` preset, whose x, y, and z axes
match the image spacing. With the reshaping handled, each
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
# A Pillow Accessor
# +++++++++++++++++
# ``to_pil`` reads an array in Pillow's layout, and ``apply`` writes the result
# of any Pillow function back to a new image with the same origin and spacing.


@pv.register_dataset_accessor('pil', pv.ImageData)
class PillowAccessor:
    """Pillow image operations for 2D images."""

    def __init__(self, image):
        self._image = image

    def to_pil(self, scalars=None):
        """Return an array of the image as a Pillow image."""
        name = scalars or self._image.active_scalars_name
        pixels = self._image.point_data.reshaped('image')[name]
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
        output.point_data.reshaped('image')[name] = np.asarray(result)
        return output

    def filter(self, image_filter, *, scalars=None):
        """Return a new image filtered by a Pillow image filter."""
        return self.apply(lambda image: image.filter(image_filter), scalars=scalars)


# %%
# Read the PyVista logo and blur it with a Pillow filter.
logo = pv.read(examples.logofile)
blurred = logo.pil.filter(ImageFilter.GaussianBlur(8))
# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore
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
# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore
pl = pv.Plotter()
pl.add_mesh(
    relief, scalars=logo.point_data['PNGImage'][:, :3], rgb=True, smooth_shading=True
)
pl.view_xy()
pl.camera.elevation = -50
pl.show()

# %%
# A SciPy Accessor
# ++++++++++++++++
# :mod:`scipy.ndimage` filters work on arrays of any dimension. With the
# ``'grid'`` preset the axes are x, y, and z, matching ``spacing``, so the filter
# sizes convert from physical units to voxels.
# :meth:`~pyvista.ArrayReshaper.wrap` turns each SciPy function into a filter,
# so each accessor method is one call.
xyz = pv.ArrayReshaper.grid()


@pv.register_dataset_accessor('ndimage', pv.ImageData)
class NdimageAccessor:
    """SciPy n-dimensional image operations."""

    def __init__(self, image):
        self._image = image

    def gaussian_filter(self, sigma, *, scalars=None):
        """Return a copy smoothed by a Gaussian with ``sigma`` in physical units."""
        voxels = np.divide(sigma, self._image.spacing)
        smooth = xyz.wrap(lambda volume: ndimage.gaussian_filter(volume, voxels))
        return smooth(self._image, scalars)

    def label(self, threshold, *, scalars=None):
        """Return a copy with connected regions above ``threshold`` labeled."""
        label = xyz.wrap(lambda volume: ndimage.label(volume > threshold)[0])
        return label(self._image, scalars, output_name='labels')


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
# Compare a slice of the noisy and the smoothed volume. With the ``'zyx'``
# preset each slice is ``array[row, column]``, which Matplotlib plots directly.
k = dimensions[2] // 2
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for ax, image, title in zip(axes, [volume, smooth], ['noisy', 'smoothed'], strict=True):
    ax.imshow(
        image.point_data.reshaped('zyx')['density'][k], origin='lower', cmap='magma'
    )
    ax.set_title(title)
    ax.set_axis_off()
plt.show()

# %%
# Show the labeled blobs in 3D.
blobs = labeled.threshold(0.5, scalars='labels', all_scalars=True)
blobs.plot(scalars='labels', cmap='tab10', show_scalar_bar=False)

# %%
# Filters on Every Block
# ++++++++++++++++++++++
# A filter made with :meth:`~pyvista.ArrayReshaper.wrap` runs on every block of
# a :class:`~pyvista.MultiBlock` through :meth:`~pyvista.MultiBlock.generic_filter`.
# Add the median of each scan to its block, in place.
median = xyz.wrap(lambda volume: ndimage.median_filter(volume, size=3))
scans = pv.MultiBlock({'noisy': volume.copy(), 'smoothed': smooth.copy()})
scans.generic_filter(median, 'density', output_name='median', inplace=True)
[round(float(np.std(block['median'])), 3) for block in scans]

# %%
# Remove the accessors when they are no longer needed.
pv.unregister_dataset_accessor('pil', pv.ImageData)
pv.unregister_dataset_accessor('ndimage', pv.ImageData)
# %%
# .. tags:: filter
