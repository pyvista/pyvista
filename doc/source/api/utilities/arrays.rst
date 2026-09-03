.. _arrays_api:

Arrays
------
Data arrays are attached to a dataset through :class:`~pyvista.DataSetAttributes`,
available as :attr:`~pyvista.DataSet.point_data`,
:attr:`~pyvista.DataSet.cell_data`, and :attr:`~pyvista.DataObject.field_data`.
The functions on this page look up those arrays by name and convert between
NumPy arrays and VTK arrays.

.. seealso::

   :ref:`pyvista_data_model`
      How points, cells, and data arrays fit together.

   :ref:`point_cell_scalars_example`
      Point data versus cell data when plotting.

   :class:`~pyvista.core.utilities.arrays.FieldAssociation`
      The association returned by :func:`~pyvista.get_array_association`.

.. currentmodule:: pyvista

Array Access
~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   cell_array
   field_array
   get_array
   get_array_association
   point_array
   set_default_active_scalars
   set_default_active_vectors

Array Conversion
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   array_from_vtkmatrix
   convert_array
   pyvista_ndarray
   vtk_points
   vtkmatrix_from_array
