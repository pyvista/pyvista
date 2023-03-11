General Utilities
-----------------
.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   utilities.VtkErrorCatcher
   utilities.axis_rotation
   utilities.is_inside_bounds
   utilities.set_error_output_file


Object Conversions or Wrapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   array_from_vtkmatrix
   cubemap
   cubemap_from_filenames
   image_to_texture
   is_pyvista_dataset
   numpy_to_texture
   pyvista_ndarray
   vtkmatrix_from_array
   wrap


File IO
~~~~~~~
.. autosummary::
   :toctree: _autosummary

   read
   read_exodus
   read_legacy
   read_texture
   save_meshio


Mesh Creation
~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   CellType
   fit_plane_to_points
   lines_from_points
   vector_poly_data
   vtk_points


Array Access
~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   cell_array
   convert_array
   field_array
   get_array
   point_array


Image Comparison and Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   compare_images


Miscellaneous
~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   Color

.. autosummary::
   :toctree: _autosummary

   ColorLike
   start_xvfb

.. autosummary::
   :toctree: _autosummary

   Report


VTK Version Information
~~~~~~~~~~~~~~~~~~~~~~~
The PyVista library is heavily dependent on VTK and provides an easy
way of getting the version of VTK in your environment.

.. code:: python

   Output the version of VTK.

   >>> import pyvista
   >>> pyvista.vtk_version_info
   VTKVersionInfo(major=9, minor=1, micro=0)

   Get the major version of VTK

   >>> pyvista.vtk_version_info.major
   9
