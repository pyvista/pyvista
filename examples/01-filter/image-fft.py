"""
.. _image_fft_example:

Fast Fourier Transform
~~~~~~~~~~~~~~~~~~~~~~

This example shows how to apply a Fast Fourier Transform (FFT) to a
:class:`pyvista.UniformGrid` using :func:`pyvista.UniformGridFilters.fft`
filter.

Here, we demonstrate FFT usage by denoising an image, effectively removing any
"high frequency" content by performing a `low pass filter
<https://en.wikipedia.org/wiki/Low-pass_filter>`_.

This example was inspired by `Image denoising by FFT
<https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html>`_.

"""

import pyvista as pv
from pyvista import examples

###############################################################################
# Load the example Moon landing image and plot it.

image = examples.download_moonlanding_image()
print(image.point_data)

# Create a theme that we can reuse when plotting the image
grey_theme = pv.themes.DocumentTheme()
grey_theme.cmap = 'gray'
grey_theme.show_scalar_bar = False
grey_theme.axes.show = False
image.plot(theme=grey_theme, cpos='xy', text='Unprocessed Moon Landing Image')

###############################################################################
# Apply FFT to the image
#
# FFT will be applied to the active scalars, which is stored in ``'PNGImage'``.
# The output from the filter contains both real and imaginary components and is
# stored in the same array.

fft_image = image.fft()
fft_image.point_data

###############################################################################
# Plot the FFT of the image. Note that we are effectively viewing the
# "frequency" of the data in this image, where the four corners contain the low
# frequency content of the image, and the middle is the high frequency content
# of the image.
#
# .. note::
#    VTK internally creates a normalized array to plot both the real and
#    imaginary values from the FFT filter. To avoid having this array included
#    in the dataset, we use :func:`copy() <pyvista.DataObject.copy>` to create a
#    temporary copy that's plotted.

fft_image.copy().plot(
    cpos="xy",
    theme=grey_theme,
    log_scale=True,
    text='Moon Landing Image FFT',
)

###############################################################################
# Remove the noise from the fft_image
#
# Effectively, we want to remove high frequency (noisy) data from our image.
# First, let's reshape by the size of the image. Note that the image data is in
# real and imaginary axes.
#
# Next, perform a low pass filter by removing the middle 80% of the content of
# the image. Note that the high frequency content is in the middle of the array.

per_keep = 0.10

width, height, _ = fft_image.dimensions
data = fft_image['PNGImage'].reshape(height, width, 2)  # note: axes flipped
data[int(height * per_keep) : -int(height * per_keep)] = 0
data[:, int(width * per_keep) : -int(width * per_keep)] = 0

fft_image.copy().plot(
    cpos="xy",
    theme=grey_theme,
    log_scale=True,
    text='Moon Landing Image FFT with Noise Removed ',
)


###############################################################################
# Finally, convert the image data back to the "image domain" and plot it.

rfft = fft_image.rfft()
rfft.plot(cpos="xy", theme=grey_theme, text='Processed Moon Landing Image')
