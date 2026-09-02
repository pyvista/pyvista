.. _image_api:

Image
-----

PyVista includes several sources for generating image data, along with
functions for sampling implicit functions and comparing images.

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

Implicit Functions
~~~~~~~~~~~~~~~~~~
An implicit function defines a scalar value at every point in space.
:func:`~pyvista.sample_function` evaluates one over a grid to produce a
:class:`~pyvista.ImageData`.

.. autosummary::
   :toctree: _autosummary

   generate_plane
   perlin_noise
   sample_function

Image Comparison
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   compare_images
