
Using Local VTK-Data
====================

Normally, the PyVista examples will be downloaded from the 
`pyvista/vtk-data repository <https://github.com/pyvista/vtk-data>`_.
Alternatively, the entire pyvista/vtk-data repository can be supplied as a local folder.
If the ``PYVISTA_VTK_DATA`` environment variable is set to the folder path, the examples will 
instead be copied from the local folder.

This example uses a cloned vtk-data repository:

.. code-block:: bash

    git clone https://github.com/pyvista/vtk-data.git path/to/repo/vtk-data  # change the path
    export PYVISTA_VTK_DATA=path/to/repo/vtk-data  # change the path
