.. _transform_api:

Transformations
---------------
The :class:`~pyvista.Transform` class describes linear transformations and is
accepted wherever a transformation is used, such as
:meth:`~pyvista.DataObjectFilters.transform`. The functions on this page
transform meshes, points, and vectors directly.

.. seealso::

   :ref:`rotate_example`
      Rotate a mesh about an axis.

   :ref:`icp_registration_example`
      Align two surfaces with an iterative closest point transform.

   :meth:`~pyvista.Prop3D.transform`
      Transform an actor instead of its mesh.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   Transform
   translate
   core.utilities.axis_rotation

Spherical Coordinates
~~~~~~~~~~~~~~~~~~~~~
.. seealso::

   :ref:`spherical_example`
      Plot data in spherical coordinates.

   :ref:`create_sphere_example`
      Build a sphere from spherical coordinates.

.. autosummary::
   :toctree: _autosummary

   cartesian_to_spherical
   spherical_to_cartesian
   grid_from_sph_coords
   transform_vectors_sph_to_cart
