.. _conversions_api:

Conversions
-----------
These functions convert between PyVista datasets and the objects of VTK and
other mesh libraries. To read or write a file, see :ref:`reader_api` and
:meth:`~pyvista.DataObject.save`.

.. seealso::

   :ref:`wrap_trimesh_example`
      Wrap ``trimesh`` and VTK objects.

   :ref:`vtk_to_pyvista_docs`
      How PyVista's interface relates to VTK's.

.. currentmodule:: pyvista

Wrapping VTK Objects
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   is_pyvista_dataset
   wrap

Meshio
~~~~~~
.. autosummary::
   :toctree: _autosummary

   from_meshio
   is_meshio_mesh
   read_meshio
   save_meshio
   to_meshio

Trimesh
~~~~~~~
.. autosummary::
   :toctree: _autosummary

   from_trimesh
   is_trimesh_mesh
   to_trimesh
