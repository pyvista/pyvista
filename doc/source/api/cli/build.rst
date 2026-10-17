.. _cli_build:

pyvista build
-------------
Command-line interface for building a VTK backend wheel from source.

``pyvista build vtk`` clones VTK, checks for the build tools, configures and
compiles it with the options PyVista's CI uses, and writes a wheel. See
:ref:`building_vtk` for the guided walk-through, the module groups, and the
developer loop.

API Reference
*************
Show the output from ``pyvista build --help``.

.. command-output:: pyvista build --help
