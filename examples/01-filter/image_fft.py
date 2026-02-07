"""
.. _image_fft_example:

Fast Fourier Transform
~~~~~~~~~~~~~~~~~~~~~~

This example shows how to apply a Fast Fourier Transform (FFT) to a
:class:`pyvista.ImageData` using :func:`pyvista.ImageDataFilters.fft`
filter.

Here, we demonstrate FFT usage by denoising an image, effectively removing any
"high frequency" content by performing a `low pass filter
<https://en.wikipedia.org/wiki/Low-pass_filter>`_.

This example was inspired by `Image denoising by FFT
<https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html>`_.

"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

# sphinx_gallery_start_ignore
# all but first and last image are black in interactive mode
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore


# %%
# Load the example Moon landing image and plot it.

image = examples.download_moonlanding_image()
print(image.point_data)

# Create a theme that we can reuse when plotting the image
grey_theme = pv.themes.DocumentTheme()
grey_theme.cmap = 'gray'
grey_theme.show_scalar_bar = False
grey_theme.axes.show = False
image.plot(theme=grey_theme, cpos='xy', text='Unprocessed Moon Landing Image')


# %%
# Apply FFT to the image
# ~~~~~~~~~~~~~~~~~~~~~~
# FFT will be applied to the active scalars, ``'PNGImage'``, the default
# scalars name when loading a PNG image.
#
# The output from the filter is a complex array stored by the same name unless
# specified using ``output_scalars_name``.

fft_image = image.fft()
fft_image.point_data


# %%
# Plot the FFT of the image
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the absolute value of the FFT of the image.
#
# Note that we are effectively viewing the "frequency" of the data in this
# image, where the four corners contain the low frequency content of the image,
# and the middle is the high frequency content of the image.

fft_image.plot(
    scalars=np.abs(fft_image.point_data['PNGImage']),
    cpos='xy',
    theme=grey_theme,
    log_scale=True,
    text='Moon Landing Image FFT',
    copy_mesh=True,  # don't overwrite scalars when plotting
)


# %%
# Remove the noise from the ``fft_image``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Effectively, we want to remove high frequency (noisy) data from our image.
# First, let's reshape by the size of the image.
#
# Next, perform a low pass filter by removing the middle 80% of the content of
# the image. Note that the high frequency content is in the middle of the array.
#
# .. note::
#    It is easier and more efficient to use the existing
#    :func:`pyvista.ImageDataFilters.low_pass` filter. This section is here
#    for demonstration purposes.

ratio_to_keep = 0.10

# modify the fft_image data
width, height, _ = fft_image.dimensions
data = fft_image['PNGImage'].reshape(height, width)  # note: axes flipped
data[int(height * ratio_to_keep) : -int(height * ratio_to_keep)] = 0
data[:, int(width * ratio_to_keep) : -int(width * ratio_to_keep)] = 0

fft_image.plot(
    scalars=np.abs(data),
    cpos='xy',
    theme=grey_theme,
    log_scale=True,
    text='Moon Landing Image FFT with Noise Removed',
    copy_mesh=True,  # don't overwrite scalars when plotting
)


# %%
# Convert to the spatial domain using reverse FFT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally, convert the image data back to the "spatial" domain and plot it.


rfft = fft_image.rfft()
rfft['PNGImage'] = np.real(rfft['PNGImage'])
rfft.plot(cpos='xy', theme=grey_theme, text='Processed Moon Landing Image')
# %%
# .. tags:: filter
