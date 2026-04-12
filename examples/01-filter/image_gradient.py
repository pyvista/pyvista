"""
.. _image_gradient_example:

Image Gradient
~~~~~~~~~~~~~~

Compute the gradient of an image using :func:`pyvista.ImageDataFilters.gradient`.

The gradient is a vector field that points in the direction of the greatest rate
of increase of the scalar field, with its magnitude representing the rate of
change.

This example is inspired by the `VTK ImageGradient example
<https://examples.vtk.org/site/Python/VisualizationAlgorithms/ImageGradient/>`_.

See also :func:`pyvista.ImageDataFilters.gradient_magnitude`.

"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

# %%
# Load the dataset and convert to float.
# We use the built-in ``download_full_head`` example dataset, which is a CT scan
# of a human head.

image = examples.download_full_head()
image['MetaImage'] = image['MetaImage'].astype(np.float64)

# %%
# Compute the 2D gradient of the image data.
# Here we compute the gradient in 2D, which produces a 2-component
# vector field (gradient in X and Y).

gradient = image.gradient(dimensionality=2)
gradient

# %%
# Visualize the gradient magnitude as a heatmap for a single slice.
# We extract the Z-slice and show the gradient magnitude.

slice_z = 22
grad_mag = image.gradient_magnitude(dimensionality=2)

plot_kwargs = dict(
    cpos='xy',
    cmap='inferno',
    scalar_bar_args={'title': 'Gradient Magnitude'},
)

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_text('Original Image (Slice 22)', font_size=10)
pl.add_mesh(
    image.extract_subset([0, 256, 0, 256, slice_z, slice_z]), cmap='gray', cpos='xy'
)
pl.subplot(0, 1)
pl.add_text('Gradient Magnitude (Slice 22)', font_size=10)
pl.add_mesh(
    grad_mag.extract_subset([0, 256, 0, 256, slice_z, slice_z]),
    **plot_kwargs,
)
pl.link_views()
pl.show()

# %%
# Compute the 3D gradient and visualize the gradient vectors.
# For 3D data, the gradient produces a 3-component vector field.

gradient_3d = image.gradient(dimensionality=3)

# Extract a slice and warp the surface by the gradient magnitude
# for a 3D visualization
grad_mag_3d = image.gradient_magnitude(dimensionality=3)
grad_mag_3d.plot(volume=True, cmap='inferno')

# %%
# .. tags:: filter
