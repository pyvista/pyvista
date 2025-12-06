"""
.. _gaussian_smoothing_example:

Gaussian Smoothing
~~~~~~~~~~~~~~~~~~

Perform a Gaussian convolution on a uniformly gridded data set.

:class:`pyvista.ImageData` data sets (a.k.a. images) a can be smoothed by
convolving the image data set with a Gaussian for one- to three-dimensional
inputs. This is commonly referred to as Gaussian blurring and typically used
to reduce noise or decrease the detail of an image dataset.

See also :func:`pyvista.ImageDataFilters.gaussian_smooth`.

"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import pyvista as pv
from pyvista import examples

# Load dataset
data = examples.download_gourds()

# Define a good point of view
cp = pv.CameraPosition(
    position=(319.5, 239.5, 1053.7372980874645),
    focal_point=(319.5, 239.5, 0.0),
    viewup=(0.0, 1.0, 0.0),
)

# %%
# Let's apply the Gaussian smoothing with different values of standard
# deviation.
pl = pv.Plotter(shape=(2, 2))

pl.subplot(0, 0)
pl.add_text('Original Image', font_size=14)
pl.add_mesh(data, rgb=True)
pl.camera_position = cp

pl.subplot(0, 1)
pl.add_text('Gaussian smoothing, std=2', font_size=14)
pl.add_mesh(data.gaussian_smooth(std_dev=2.0), rgb=True)
pl.camera_position = cp

pl.subplot(1, 0)
pl.add_text('Gaussian smoothing, std=4', font_size=14)
pl.add_mesh(data.gaussian_smooth(std_dev=4.0), rgb=True)
pl.camera_position = cp

pl.subplot(1, 1)
pl.add_text('Gaussian smoothing, std=8', font_size=14)
pl.add_mesh(data.gaussian_smooth(std_dev=8.0), rgb=True)
pl.camera_position = cp

pl.show()

# %%
# |
#
# Volume Rendering
# ~~~~~~~~~~~~~~~~
# Now let's see an example on a 3D dataset with volume rendering:

# sphinx_gallery_start_ignore
# volume rendering does not work in interactive plots currently
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

data = examples.download_brain()

smoothed_data = data.gaussian_smooth(std_dev=3.0)


dargs = dict(clim=smoothed_data.get_data_range(), opacity=[0, 0, 0, 0.1, 0.3, 0.6, 1])

n = [100, 150, 200, 245, 255]

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_text('Original Image', font_size=24)
# pl.add_mesh(data.contour(n), **dargs)
pl.add_volume(data, **dargs)
pl.subplot(0, 1)
pl.add_text('Gaussian smoothing', font_size=24)
# pl.add_mesh(smoothed_data.contour(n), **dargs)
pl.add_volume(smoothed_data, **dargs)
pl.link_views()
pl.camera_position = pv.CameraPosition(
    position=(-162.0, 704.8, 65.02),
    focal_point=(90.0, 108.0, 90.0),
    viewup=(0.0068, 0.0447, 0.999),
)
pl.show()
# %%
# .. tags:: filter
