.. _conversions_api:

Conversions
-----------
These functions convert between PyVista datasets and the objects of VTK and
other mesh libraries.

.. currentmodule:: pyvista

Wrapping VTK Objects
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   wrap
   is_pyvista_dataset

Meshio
~~~~~~
.. autosummary::
   :toctree: _autosummary

   from_meshio
   to_meshio
   is_meshio_mesh
   read_meshio
   save_meshio

Trimesh
~~~~~~~
.. autosummary::
   :toctree: _autosummary

   from_trimesh
   to_trimesh
   is_trimesh_mesh
