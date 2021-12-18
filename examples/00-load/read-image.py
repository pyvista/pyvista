"""
Read Image Files
~~~~~~~~~~~~~~~~

Read and plot image files (JPEG, TIFF, PNG, etc).

"""
import pyvista as pv
from pyvista import examples

###############################################################################
# PyVista fully supports reading images into their own spatially referenced
# data objects (this example) as well as supports texture mapping of images
# onto datasets (see :ref:`ref_texture_example`).
#
# Download a JPEG image of a puppy and load it to :class:`pyvista.UniformGrid`.
# This could similarly be implemented with any image file by using the
# :func:`pyvista.read` function and passing the path to the image file.

image = examples.download_puppy()
# or...
# image = pv.read('my_image.jpg')

###############################################################################
# When plotting images stored in :class:`pyvista.UniformGrid` objects, it is
# important to specify using the `rgb` option when plotting to ensure that the
# image's true colors are used and not mapped.

# True image colors
image.plot(rgb=True, cpos="xy")

###############################################################################

# Mapped image colors
image.plot(cpos="xy")

###############################################################################
# Convert rgb to grayscale.
# https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems

r = image["JPEGImage"][:, 0]
g = image["JPEGImage"][:, 1]
b = image["JPEGImage"][:, 2]
image.clear_data()
image["GrayScale"] = 0.299 * r + 0.587 * g + 0.114 * b
pv.global_theme.cmap = "gray"
image.copy().plot(cpos="xy")

###############################################################################
# It is also possible to apply filters to images. The following is the Fast
# Fourier Transformed image data.

fft = image.image_fft()
fft.copy().plot(cpos="xy", log_scale=True)

###############################################################################
# Once Fast Fourier Transformed, images can also Reverse Fast Fourier
# Transformed.

rfft = fft.image_rfft()
rfft.copy().plot(cpos="xy")
