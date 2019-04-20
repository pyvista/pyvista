"""
Read Image Files
~~~~~~~~~~~~~~~~

Read and plot image files (JPEG, TIFF, PNG, etc).

"""

from vtki import examples

################################################################################
# ``vtki`` fully supportes reading images into their own spatially referenced
# data objects (this example) as well as supports texture mapping of images onto
# datasets (see :ref:`ref_texture_example`).
#
# Download a JPEG image of a puppy and load it to :class:`vtki.UniformGrid`.
# This could similarly be implemented with any image file by using the
# :func:`vtki.read` function and passing the path to the image file.

image = examples.download_puppy()
# or...
# image = vtki.read('my_image.jpg')

################################################################################
# When plotting images stored in :class:`vtki.UniformGrid` objects, it is
# important to specify usign the `rgb` option when plotting to ensure that the
# image's true colors are used and not mapped.

# True image colors
image.plot(rgb=True, cpos='xy')

################################################################################

# Mapped image colors
image.plot(cpos='xy')
