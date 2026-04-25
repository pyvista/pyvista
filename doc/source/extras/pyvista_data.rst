
Using Local pyvista/data
========================

Normally, the PyVista examples will be downloaded from the
`pyvista/data repository <https://github.com/pyvista/data>`_.
Alternatively, the entire pyvista/data repository can be supplied as a local folder.
If the ``PYVISTA_VTK_DATA`` environment variable is set to the folder path, the examples will
instead be copied from the local folder.

This example uses a cloned ``pyvista/data`` repository:

.. code-block:: bash

    git clone https://github.com/pyvista/data.git path/to/pyvista/data  # change the path
    export PYVISTA_VTK_DATA=path/to/pyvista/data  # change the path
