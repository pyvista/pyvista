.. _arrays_api:

Arrays
------
Data arrays are attached to a dataset through :class:`~pyvista.DataSetAttributes`,
available as :attr:`~pyvista.DataSet.point_data`,
:attr:`~pyvista.DataSet.cell_data`, and :attr:`~pyvista.DataObject.field_data`.
The functions on this page look up those arrays by name and convert between
NumPy arrays and VTK arrays.

.. currentmodule:: pyvista

Array Access
~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   cell_array
   field_array
   point_array
   get_array
   get_array_association
   set_default_active_scalars
   set_default_active_vectors

Array Conversion
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   pyvista_ndarray
   convert_array
   array_from_vtkmatrix
   vtkmatrix_from_array
   vtk_points
