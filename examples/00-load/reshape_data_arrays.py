"""
.. _reshape_data_arrays_example:

Reshape Data Arrays
~~~~~~~~~~~~~~~~~~~

Get and set data arrays as shaped NumPy arrays instead of flat arrays.

PyVista stores every data array flat, with one row per point or cell.
``point_data.reshaped`` and ``cell_data.reshaped`` show the same arrays in a
shaped layout chosen by :attr:`pyvista.global_config.array_reshaping
<pyvista.core.config.Config.array_reshaping>` or by
:meth:`~pyvista.core._array_reshaping.ReshapedArrays.using`.

"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageSequence
import pyvista as pv
from pyvista import examples

# %%
# How Arrays Are Indexed in Space
# +++++++++++++++++++++++++++++++
# A NumPy image is indexed ``array[row, column]``. Matplotlib draws row 0 at
# the top, rows run down and columns run right.
pixels = np.arange(12).reshape(3, 4)
pixels

# %%
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(pixels, cmap='PuBu')
for (row, column), value in np.ndenumerate(pixels):
    ax.text(column, row, f'[{row}, {column}]\nvalue {value}', ha='center', va='center')
ax.set_xlabel('column')
ax.set_ylabel('row')
ax.set_title('NumPy array in Matplotlib')
plt.show()

# %%
# VTK stores the same image flat, one value per pixel. Here each pixel is a
# cell centered on integer x and y. Pixel 0 is at the origin in the bottom
# left corner, and the ids run along x first and then up along y. The
# ``'image'`` preset puts the top row of the NumPy array at the largest y, so
# the picture stays upright.
image = pv.ImageData(dimensions=(5, 4, 1), origin=(-0.5, -0.5, 0))
image.cell_data.reshaped.using('image')['value'] = pixels
image.cell_data['value']

# %%
labels = [f'id {cell_id}\nvalue {value}' for cell_id, value in enumerate(image['value'])]
# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore
pl = pv.Plotter()
pl.add_mesh(
    image,
    scalars='value',
    cmap='PuBu',
    show_edges=True,
    lighting=False,
    show_scalar_bar=False,
)
pl.add_point_labels(
    image.cell_centers().points,
    labels,
    font_size=14,
    show_points=False,
    always_visible=True,
)
pl.show_bounds(xtitle='x', ytitle='y', n_xlabels=5, n_ylabels=4, fmt='%.1f')
pl.view_xy()
pl.show()

# %%
# Each preset is a different index order on the same stored values.
# ``'image'`` gives back the NumPy array.
image.cell_data.reshaped.using('image')['value']

# %%
# ``'grid'`` indexes the same values as ``array[i, j, k]`` along x, y, and z,
# so ``array[0, 0, 0]`` is pixel 0 at the origin, and ``array[i, j, 0]`` is
# the pixel at ``x = i`` and ``y = j``.
image.cell_data.reshaped['value'][:, :, 0]

# %%
# ``'zyx'`` indexes them as ``array[k, j, i]``, which is the NumPy image
# upside down: pixel 0 is in the first row.
image.cell_data.reshaped.using('zyx')['value'][0]

# %%
# Compare With Matplotlib
# +++++++++++++++++++++++
# Sample a 2D field on a grid of 60 by 30 points and plot it with Matplotlib.
xx, yy = np.meshgrid(np.linspace(0, 2, 60), np.linspace(0, 1, 30), indexing='ij')
field = np.sin(2 * np.pi * xx) * yy
field.shape

# %%
fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, field, cmap='viridis')
ax.set_aspect('equal')
plt.show()

# %%
# Setting the field through ``reshaped`` stores it in the point order of the
# grid, so PyVista shows the same picture. Flattening it by hand with
# :meth:`numpy.ndarray.ravel` uses C order instead, and scrambles it.
plane = pv.ImageData(dimensions=(60, 30, 1), spacing=(2 / 59, 1 / 29, 1))
plane.point_data.reshaped['reshaped'] = field
plane.point_data['ravel'] = field.ravel()

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore
pl = pv.Plotter(shape=(1, 2), window_size=(1000, 300))
for index, name in enumerate(['reshaped', 'ravel']):
    pl.subplot(0, index)
    pl.add_text(name, font_size=12)
    pl.add_mesh(plane.copy(), scalars=name, cmap='viridis', show_scalar_bar=False)
    pl.view_xy()
pl.show()

# %%
# Getting the field back gives the grid shape, which Matplotlib plots directly.
fig, ax = plt.subplots()
ax.pcolormesh(xx, yy, plane.point_data.reshaped['reshaped'][:, :, 0], cmap='viridis')
ax.set_aspect('equal')
plt.show()

# %%
# Volumes in XYZ Order
# +++++++++++++++++++++
# The default ``'grid'`` preset matches :func:`numpy.meshgrid` with
# ``indexing='ij'``: ``array[i, j, k]`` is the value at index ``i`` along x,
# ``j`` along y, and ``k`` along z, the order of
# :attr:`~pyvista.core._structured_data._StructuredDataMixin.dimensions`.
# The same layout is used by ``nibabel`` volumes.
x, y, z = np.meshgrid(
    np.linspace(-1, 1, 40), np.linspace(-1, 1, 30), np.linspace(-1, 1, 20), indexing='ij'
)
values = np.sin(3 * x) * np.cos(3 * y) + z

grid = pv.ImageData(dimensions=values.shape, spacing=(0.05, 0.05, 0.1))
grid.point_data.reshaped['values'] = values
grid.point_data.reshaped['values'].shape

# %%
# The stored array is flat, so every filter and plot uses it as usual.
grid.point_data['values'].shape

# %%
grid.contour(5).plot(cmap='viridis')

# %%
# Cell data takes the shape of the cells, one less than the points along each
# axis.
grid.cell_data.reshaped['cell_values'] = values[:-1, :-1, :-1]
grid.cell_data.reshaped['cell_values'].shape

# %%
# The reshaped arrays are views of the stored arrays, so editing them in place
# edits the dataset.
grid.point_data.reshaped['values'][0, 0, 0] = 10.0
grid.point_data['values'][0]

# %%
# Vector Fields
# +++++++++++++
# Multi-component arrays keep their components on the last axis.
grid.point_data['coordinates'] = grid.points
grid.point_data.reshaped['coordinates'].shape

# %%
# :func:`numpy.gradient` returns one array per axis, which stack to a
# components-first array. The ``'channels_first'`` preset reads and writes that
# layout.
gradient = np.stack(np.gradient(values, *grid.spacing))
gradient.shape

# %%
grid.point_data.reshaped.using('channels_first')['gradient'] = gradient
grid.point_data['gradient'].shape

# %%
glyphs = grid.extract_subset((0, 39, 0, 29, 10, 10), rate=(3, 3, 1)).glyph(
    orient='gradient', scale=False, factor=0.08
)
glyphs.plot(cmap='coolwarm', cpos='xy')

# %%
# Volumes in ZYX Order
# +++++++++++++++++++++
# Many imaging libraries index volumes the other way round, as
# ``array[k, j, i]``: ``SimpleITK.GetArrayFromImage``, scikit-image,
# ``tifffile`` and image stacks read slice by slice. The ``'zyx'`` preset
# reads and writes that layout without copying.
zyx_volume = values.transpose(2, 1, 0)
zyx_volume.shape

# %%
volume = pv.ImageData(dimensions=grid.dimensions)
volume.point_data.reshaped.using('zyx')['values'] = zyx_volume
np.array_equal(volume.point_data.reshaped['values'], values)

# %%
# Scans With Several Modalities
# +++++++++++++++++++++++++++++
# Deep learning frameworks such as PyTorch and MONAI store scans with several modalities
# channels first, as ``(channel, z, y, x)``. Build a reshaper for that layout
# with :class:`~pyvista.GridReshaper`. Here two synthetic modalities stand in
# for T1-weighted and T2-weighted MRI.
t1 = np.exp(-(x**2 + y**2 + z**2) * 3)
t2 = 1 - t1
scan = np.stack([t1, t2]).transpose(0, 3, 2, 1)
scan.shape

# %%
channels_first_zyx = pv.GridReshaper(axes='zyx', components='first')
mri = pv.ImageData(dimensions=grid.dimensions)
mri.point_data.reshaped.using(channels_first_zyx)['scan'] = scan
mri.point_data['scan'].shape

# %%
# Split the modalities into separate arrays to plot them one at a time.
mri.point_data['T1'] = mri.point_data['scan'][:, 0]
mri.point_data['T2'] = mri.point_data['scan'][:, 1]
mri.slice_orthogonal().plot(scalars='T1', cmap='bone')

# %%
# 2D Images
# +++++++++
# Pillow, OpenCV, imageio and :func:`matplotlib.pyplot.imread` return images
# as ``(row, column, channel)`` arrays whose first row is the top of the
# image. The ``'image'`` preset reads and writes that layout. It indexes the
# image as ``array[y, x]``, drops the flat z axis, and flips y so that the
# image is upright.
rgba = np.asarray(Image.open(examples.logofile))
rgba.shape

# %%
height, width = rgba.shape[:2]
logo = pv.ImageData(dimensions=(width, height, 1))
logo.point_data.reshaped.using('image')['rgba'] = rgba
logo.point_data['rgba'].shape

# %%
# This is the same array PyVista reads from the file.
np.array_equal(logo.point_data['rgba'], pv.read(examples.logofile).point_data['PNGImage'])

# %%
logo.plot(rgb=True, cpos='xy', background='white')

# %%
# Custom Reshapers
# ++++++++++++++++
# Any layout can be read and written with an :class:`~pyvista.ArrayReshaper`
# built from two functions. OpenCV returns color images as BGR rather than RGB,
# so reverse the channels on top of the ``'image'`` preset.
image_preset = pv.GridReshaper(axes='zyx', squeeze=True, origin='upper')


def bgr_to_grid(array, *, dataset, association):
    """Return an RGB image from PyVista as a BGR image."""
    return image_preset.reshape(array, dataset=dataset, association=association)[
        ..., ::-1
    ]


def grid_to_bgr(array, *, dataset, association):
    """Return a BGR image as an RGB array for PyVista."""
    return image_preset.flatten(
        array[..., ::-1], dataset=dataset, association=association
    )


opencv = pv.ArrayReshaper(reshape=bgr_to_grid, flatten=grid_to_bgr, name='opencv')

bgr = rgba[..., 2::-1]  # an image as returned by cv2.imread
logo.point_data.reshaped.using(opencv)['from_opencv'] = bgr
np.array_equal(logo.point_data['from_opencv'], logo.point_data['rgba'][:, :3])

# %%
# Animation Frames
# ++++++++++++++++
# A stack of frames, as ``(frame, row, column, channel)``, is a volume with
# time along z. The ``'image'`` preset reads and writes it the same way as a
# single image.
filename = examples.download_gif_simple(load=False)
with Image.open(filename) as gif:
    frames = np.stack(
        [np.asarray(frame.convert('RGB')) for frame in ImageSequence.Iterator(gif)]
    )
frames.shape

# %%
n_frames, height, width = frames.shape[:3]
movie = pv.ImageData(dimensions=(width, height, n_frames), spacing=(1, 1, 50))
movie.point_data.reshaped.using('image')['frames'] = frames
movie.point_data.reshaped.using('image')['frames'].shape

# %%
movie.slice_along_axis(n_frames, axis='z').plot(rgb=True, background='white')

# %%
# Tensors
# +++++++
# The ``'tensor'`` preset shows nine-component arrays as 3x3 matrices on any
# dataset, matching the tensor layout VTK uses.
sphere = pv.Sphere()
normals = sphere.point_data['Normals']
sphere.point_data.reshaped.using('tensor')['outer'] = (
    normals[:, :, None] * normals[:, None, :]
)
sphere.point_data['outer'].shape

# %%
sphere.point_data.reshaped.using('tensor')['outer'].shape

# %%
# Changing the Default
# ++++++++++++++++++++
# Set :attr:`pyvista.global_config.array_reshaping
# <pyvista.core.config.Config.array_reshaping>` to use a layout everywhere.
pv.global_config.array_reshaping = 'zyx'
grid.point_data.reshaped['values'].shape

# %%
pv.global_config.array_reshaping = 'grid'  # restore the default
# %%
# .. tags:: load
