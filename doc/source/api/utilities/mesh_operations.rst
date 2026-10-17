.. _mesh_operations_api:

Mesh Operations
---------------
These functions build a mesh from arrays or operate on a dataset as a whole.
Most operations on a mesh are available as filter methods; see :ref:`filters`.

.. currentmodule:: pyvista

Mesh Creation
~~~~~~~~~~~~~
These functions build a mesh from existing points and faces, or from other
datasets.

.. autosummary::
   :toctree: _autosummary

   create_grid
   line_segments_from_points
   lines_from_points
   make_tri_mesh
   merge
   vector_poly_data

.. seealso::

   :ref:`create_spline_example`
      Build a spline from an array of points.

   :meth:`~pyvista.DataSetFilters.merge`
      Merge datasets as a method on the dataset.

Points
~~~~~~
These functions operate on arrays of points.

.. autosummary::
   :toctree: _autosummary

   fit_line_to_points
   fit_plane_to_points
   core.utilities.is_inside_bounds
   principal_axes

.. seealso::

   :ref:`point_cloud_orientation_example`
      Fit a plane and a line to a point cloud using its principal axes.

Implicit Functions
~~~~~~~~~~~~~~~~~~
An implicit function defines a scalar value at every point in space, such as
the signed distance from a plane. :func:`~pyvista.sample_function` evaluates
one over a grid to produce a :class:`~pyvista.ImageData`.

.. autosummary::
   :toctree: _autosummary

   generate_plane
   perlin_noise
   sample_function

.. seealso::

   :ref:`perlin_noise_2d_example`
      Sample Perlin noise in 2D.

   :ref:`perlin_noise_3d_example`
      Sample Perlin noise in 3D.

   :ref:`image_fft_perlin_noise_example`
      Filter sampled noise with a fast Fourier transform.
