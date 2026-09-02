.. _image_api:

Image
-----

PyVista includes several sources for generating image data, along with a
function for comparing images.

.. currentmodule:: pyvista

Image Sources
~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   ImageEllipsoidSource
   ImageGaussianSource
   ImageGridSource
   ImageMandelbrotSource
   ImageNoiseSource
   ImageSinusoidSource

Image Comparison
~~~~~~~~~~~~~~~~
.. seealso::

   :ref:`cli_compare`
      Compare two files from the command line.

.. autosummary::
   :toctree: _autosummary

   compare_images
