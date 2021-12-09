"""
Read Image Files
~~~~~~~~~~~~~~~~

Read and plot image files (JPEG, TIFF, PNG, etc).

"""
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
# It is also possible to apply filters to images.

fft = image.fft()
rfft = image.rfft()
pl = pv.Plotter(1, 3)
pl.subplot(0, 0)
pl.add_title("Original")
pl.add_mesh(image)
pl.subplot(0, 1)
pl.add_title("FFT")
pl.add_mesh(fft)
pl.subplot(0, 2)
pl.add_title("rFFT")
pl.add_mesh(rfft)
pl.show(cpos="xy")
