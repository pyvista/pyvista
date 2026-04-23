.. _helpers_api:

Helpers
=======

The ``pyvista`` module contains several functions to simplify the
creation and manipulation of meshes or interfacing with VTK datasets.

See :ref:`utilities-api-index` for the full API reference of all
utility functions organized by category.

.. currentmodule:: pyvista

Mesh Creation
~~~~~~~~~~~~~

.. autosummary::

   wrap
   make_tri_mesh
   lines_from_points
   line_segments_from_points
   vector_poly_data
   vtk_points

Mesh Operations
~~~~~~~~~~~~~~~

.. autosummary::

   merge
   translate
   generate_plane
   fit_plane_to_points
   fit_line_to_points

Array Utilities
~~~~~~~~~~~~~~~

.. autosummary::

   convert_array
   sample_function
   perlin_noise

Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyvista.core.config.Config
   :members:
