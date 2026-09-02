.. _transform_api:

Transformations
---------------
The :class:`~pyvista.Transform` class describes linear transformations and is
accepted wherever a transformation is used, such as
:meth:`~pyvista.DataObjectFilters.transform`. The functions on this page
transform meshes, points, and vectors directly.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   Transform
   translate
   core.utilities.axis_rotation

Spherical Coordinates
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   cartesian_to_spherical
   spherical_to_cartesian
   grid_from_sph_coords
   transform_vectors_sph_to_cart
