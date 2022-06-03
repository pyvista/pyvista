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
