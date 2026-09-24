"""
.. _resampling_example:

Detailed Resampling
~~~~~~~~~~~~~~~~~~~

This example uses :func:`pyvista.DataObjectFilters.sample`.

:func:`pyvista.DataObjectFilters.resample_to_image` samples onto a new
:class:`~pyvista.ImageData` in a single call.

:func:`pyvista.DataSetFilters.interpolate` is similar, and the two
methods are compared in :ref:`interpolate_sample_example`.

Resample one mesh's point/cell arrays onto another mesh's nodes.

"""

# %%
# This example will resample a volumetric mesh's scalar data onto the surface
# of a sphere contained in that volume.

# sphinx_gallery_thumbnail_number = 3
import pyvista as pv
from pyvista import examples

# %%
# Simple Resample
# +++++++++++++++
# Query a grid's points onto a sphere
mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
data_to_probe = examples.load_uniform()

# %%
# Plot the two datasets
pl = pv.Plotter()
pl.add_mesh(mesh, color=True)
pl.add_mesh(data_to_probe, opacity=0.5)
pl.show()

# %%
# Run the algorithm and plot the result
result = mesh.sample(data_to_probe)

# Plot result
name = 'Spatial Point Data'
result.plot(scalars=name, clim=data_to_probe.get_data_range(name))

# %%
# :func:`~pyvista.DataObjectFilters.resample_to_image` samples onto a new
# :class:`~pyvista.ImageData` instead. Pass a ``reference_volume`` to give the
# output the geometry of an image which already exists.
mesh.resample_to_image(reference_volume=data_to_probe)

# %%
# Complex Resample
# ++++++++++++++++
# Take a volume of data and create a grid of lower resolution to resample on
data_to_probe = examples.download_embryo()
mesh = pv.create_grid(data_to_probe, dimensions=(75, 75, 75))

result = mesh.sample(data_to_probe)

# %%
# :func:`~pyvista.DataObjectFilters.resample_to_image` does both steps in one call.
data_to_probe.resample_to_image(dimensions=(75, 75, 75))

# %%
# To resample :class:`~pyvista.ImageData` directly, use
# :meth:`~pyvista.ImageDataFilters.resample` instead.
data_to_probe.resample(dimensions=(75, 75, 75))

# %%
# Both meshes here are :class:`~pyvista.ImageData`, and for that case
# :meth:`~pyvista.ImageDataFilters.reslice` is the closer fit than ``sample``. It reads
# the image at the grid's points just as ``sample`` does, but returns only the resampled
# array, where ``sample`` also carries the grid's own arrays and the mask arrays probing
# adds, and it offers the border, interpolation, and anti-aliasing options an image
# needs. See :ref:`reslice_example`.
data_to_probe.reslice(mesh)

# %%
threshold = lambda m: m.threshold(75.0, scalars='SLCImage')
cpos = pv.CameraPosition(
    position=(468.9, -152.8, 152.1),
    focal_point=(121.7, 140.3, 112.3),
    viewup=(-0.1088, 0.006229, 0.994),
)
dargs = dict(clim=[0, 200], cmap='rainbow')

pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(threshold(data_to_probe), **dargs)
pl.subplot(0, 1)
pl.add_mesh(threshold(result), **dargs)
pl.link_views()
pl.view_isometric()
pl.show(cpos=cpos)

# %%
# Resample a Processed Volume
# +++++++++++++++++++++++++++
# Clipping an image returns an :class:`~pyvista.UnstructuredGrid`, which image filters do
# not accept. :func:`~pyvista.DataObjectFilters.resample_to_image` puts the processed
# volume back on a regular grid.
#
# Start from a volumetric scan of a knee. Bone is the bright end of its intensity range,
# from 100 up.
knee = examples.download_knee_full()

# sphinx_gallery_start_ignore
# the interactive scene of this volume exceeds the file size limit
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

knee.plot(volume=True, cmap='bone', clim=[100, 174])

# %%
# Clip the scan to the bone, and keep the three largest pieces: the tibia, the femur, and
# the patella. :func:`~pyvista.DataSetFilters.connectivity` numbers the regions from the
# largest down and stores the numbers as ``'RegionId'``.
bone = knee.clip_scalar(scalars='SLCImage', value=100, invert=False)
bones = bone.connectivity('specified', [0, 1, 2])
bones.plot(scalars='RegionId', cmap='glasbey', categories=True)

# %%
# Resample the labeled bones onto the scan's own grid with ``reference_volume``, so that
# a voxel of the output is a voxel of the scan. Sample the labels as categories to keep
# them whole, and blank the voxels which fall outside the bone.
labels = bones.resample_to_image(reference_volume=knee, categorical=True, mark_blank=True)

# %%
# The labels sit on the points of the image, so render them as voxel cells with
# :func:`~pyvista.ImageDataFilters.points_to_cells`.
labels.points_to_cells().plot(scalars='RegionId', cmap='glasbey', categories=True)

# %%
# .. tags:: filter
