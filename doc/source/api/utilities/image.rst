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

.. seealso::

   :ref:`perlin_noise_2d_example`
      Sample Perlin noise in 2D.

   :ref:`perlin_noise_3d_example`
      Sample Perlin noise in 3D.

   :ref:`image_fft_perlin_noise_example`
      Filter sampled noise with a fast Fourier transform.

.. autosummary::
   :toctree: _autosummary

   generate_plane
   perlin_noise
   sample_function

Image Comparison
~~~~~~~~~~~~~~~~
.. seealso::

   :ref:`cli_compare`
      Compare two files from the command line.

.. autosummary::
   :toctree: _autosummary

   compare_images
