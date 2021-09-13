General Utilities
-----------------
.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   utilities.VtkErrorCatcher
   utilities.set_error_output_file
   utilities.is_inside_bounds


Object Conversions
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   wrap
   is_pyvista_dataset
   image_to_texture
   numpy_to_texture
   array_from_vtkmatrix
   vtkmatrix_from_array
   cubemap


File IO
~~~~~~~
.. autosummary::
   :toctree: _autosummary

   read
   read_exodus
   read_texture
   read_legacy
   save_meshio


Mesh Creation
~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   lines_from_points
   vtk_points
   vector_poly_data
   fit_plane_to_points


Array Access
~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   get_array
   convert_array
   point_array
   cell_array
   field_array


Image Comparison and Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   compare_images


Miscellaneous
~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   start_xvfb
