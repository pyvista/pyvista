General Utilities
-----------------
.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   core.utilities.VtkErrorCatcher
   core.utilities.axis_rotation
   core.utilities.is_inside_bounds
   core.utilities.set_error_output_file


Object Conversions or Wrapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   array_from_vtkmatrix
   cubemap
   cubemap_from_filenames
   from_meshio
   from_trimesh
   image_to_texture
   is_meshio_mesh
   is_trimesh_mesh
   is_pyvista_dataset
   numpy_to_texture
   pyvista_ndarray
   to_meshio
   to_trimesh
   vtkmatrix_from_array
   wrap

Features
~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   cartesian_to_spherical
   cell_quality_info
   create_grid
   grid_from_sph_coords
   merge
   perlin_noise
   principal_axes
   sample_function
   spherical_to_cartesian
   transform_vectors_sph_to_cart
   voxelize
   voxelize_volume

File IO
~~~~~~~
.. autosummary::
   :toctree: _autosummary

   get_ext
   read
   read_exodus
   read_grdecl
   read_meshio
   read_texture
   save_meshio
   set_pickle_format

Mesh Creation
~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   fit_line_to_points
   fit_plane_to_points
   lines_from_points
   vector_poly_data
   vtk_points

Enums
~~~~~
.. autosummary::
   :toctree: _autosummary

   CellType
   CellStatus

Array Access
~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   cell_array
   convert_array
   field_array
   get_array
   point_array


Transformations
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   Transform

Image Comparison and Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   compare_images


Colors
~~~~~~
.. autosummary::
   :toctree: _autosummary

   Color
   ColorLike

Named colors supported by :class:`~pyvista.Color`, :class:`~pyvista.Plotter`,
and other plotting-related methods:

.. toctree::
   :maxdepth: 3

   /api/utilities/named_colors


Named colormaps supported by :class:`~pyvista.LookupTable`, :class:`~pyvista.Plotter`,
and other plotting-related methods:

.. toctree::
   :maxdepth: 3

   /api/utilities/named_colormaps

Miscellaneous
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   set_new_attribute
   allow_new_attributes
   start_xvfb
   Report
   vtk_verbosity
   vtk_snake_case

PyVista Version Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The PyVista library provides a way of getting the version installed in your
environment.

>>> # Output the version of PyVista.
>>> import pyvista as pv
>>> pv.version_info
(0, 44, 0)

VTK Version Information
~~~~~~~~~~~~~~~~~~~~~~~
The PyVista library is heavily dependent on VTK and provides an easy
way of getting the version of VTK in your environment.

>>> # Output the version of VTK.
>>> import pyvista as pv
>>> pv.vtk_version_info
VTKVersionInfo(major=9, minor=1, micro=0)

>>> # Get the major version of VTK
>>> pv.vtk_version_info.major
9
