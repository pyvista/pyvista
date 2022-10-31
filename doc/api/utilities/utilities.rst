General Utilities
-----------------
.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   utilities.VtkErrorCatcher
   utilities.set_error_output_file
   utilities.is_inside_bounds
   utilities.axis_rotation


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
   cubemap_from_filenames


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

   Color

.. autosummary::
   :toctree: _autosummary

   color_like
   start_xvfb


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
