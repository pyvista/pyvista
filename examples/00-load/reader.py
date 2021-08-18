"""
Load data using a Reader
~~~~~~~~~~~~~~~~~~~~~~~~

"""
###############################################################################
# To have more control over reading data files, use a class based reader.
# This class allows for more fine-grained control over reading datasets from
# files.  See :func:`pyvista.get_reader` for a list of file types supported.

import pyvista
from pyvista import examples
import numpy as np
from tempfile import NamedTemporaryFile

###############################################################################
# An XML PolyData file in ``.vtp`` format is created.  It will be saved in a
# temporary file for this example.

temp_file = NamedTemporaryFile('w', suffix=".vtp")
temp_file.name

###############################################################################
# :class:`pyvista.Sphere` already includes ``Normals`` point data.  Additionally
# ``height`` point data and ``id`` cell data is added.
mesh = pyvista.Sphere()
mesh['height'] = mesh.points[:, 1]
mesh['id'] = np.arange(mesh.n_cells)
mesh.save(temp_file.name)

###############################################################################
# :func:`pyvista.read` function reads all the data in the file. This provides
# a quick and easy one-liner to read data from files.

new_mesh = pyvista.read(temp_file.name)
print(f"All arrays: {mesh.array_names}")

###############################################################################
# Using :func:`pyvista.get_reader` enables more fine-grained control of reading data
# files. Reading in a ``.vtp``` file uses the :class:`pyvista.XMLPolyDataReader`.

reader = pyvista.get_reader(temp_file.name)
reader
# Alternative method: reader = pyvista.XMLPolyDataReader(temp_file.name)

###############################################################################
# Some reader classes, including this one, offer the ability to inspect the
# data file before loading all the data. For example, we can access the number
# and names of point and cell arrays.

print(f"Number of point arrays: {reader.number_point_arrays}")
print(f"Available point data:   {reader.point_array_names}")
print(f"Number of cell arrays:  {reader.number_cell_arrays}")
print(f"Available cell data:    {reader.cell_array_names}")

###############################################################################
# We can select which data to read by selectively disabling or enabling
# specific arrays or all arrays.  Here we disable all the cell arrays and
# the ``Normals`` point array to leave only the ``height`` point array.  The data
# is finally read into a pyvista object that only has the ``height`` point array.

reader.disable_all_cell_arrays()
reader.disable_point_array('Normals')
print(f"Point array status: {reader.all_point_arrays_status}")
print(f"Cell array status:  {reader.all_cell_arrays_status}")
reader_mesh = reader.read()
print(f"Read arrays:        {reader_mesh.array_names}")

###############################################################################
# We can reuse the reader object to choose different variables if needed.

reader.enable_all_cell_arrays()
reader_mesh_2 = reader.read()
print(f"New read arrays: {reader_mesh_2.array_names}")
