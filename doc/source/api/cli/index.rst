.. _cli_api:

Command Line Interface
======================

PyVista ships a command-line interface (CLI) for plotting, converting, and validating
mesh files, as well as generating environment reports, without requiring Python code.
The CLI is enabled with a standard PyVista installation, e.g.:

.. code-block:: bash

    pip install pyvista

The CLI can be invoked either with the ``pyvista`` command or as a Python module.
For example,

.. code-block:: bash

    pyvista --help

is equivalent to

.. code-block:: bash

    python -m pyvista --help

Show the output from ``pyvista --help`` to show all available subcommands and arguments.

.. command-output:: pyvista --help

Each subcommand is documented below.

.. _cli_convert:

Convert
-------
The ``pyvista convert`` command is similar to calling :func:`~pyvista.read` and
:func:`~pyvista.DataObject.save` together in Python.

Examples
********
Convert a PLY file (``.ply``) into a VTK XML PolyData file (``.vtp``).

.. command-output:: pyvista convert ant.ply ant_converted.vtp
   :cwd: /_local_examples

Specify the extension only to keep the same file name.

.. command-output:: pyvista convert ant.ply .vtp
   :cwd: /_local_examples

Use wildcard patterns to convert multiple files at once into an explicit output directory.

.. command-output:: pyvista convert *.ply output/.vtp
   :cwd: /_local_examples

API Reference
*************
Show the output from ``pyvista convert --help``.

.. command-output:: pyvista convert --help

.. _cli_plot:

Plot
----
The ``pyvista plot`` command is similar to calling :func:`~pyvista.plot` in Python.

Examples
********
Plot a single mesh file.

.. command-output:: pyvista plot ant.ply
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import pyvista as pv
   from pathlib import Path
   pv.plot(Path('source') / '_local_examples' / 'ant.ply')

Plot multiple PLY mesh files in a single window using wildcard patterns and use ``--zoom``.

.. command-output:: pyvista plot *.ply --zoom 2
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import pyvista as pv
   from pathlib import Path
   pv.plot(list((Path('source') / '_local_examples').glob('*.ply')), zoom=2)

Plot a mesh file off-screen and save a screenshot.

.. command-output:: pyvista plot ant.ply --screenshot output.png --off-screen
   :cwd: /_local_examples

API Reference
*************
Show the output from ``pyvista plot --help``.

.. command-output:: pyvista plot --help

.. _cli_report:

Report
------
The ``pyvista report`` command is similar to creating a :class:`~pyvista.Report` in Python.

Examples
********
Show a PyVista report.

.. command-output:: pyvista report

API Reference
*************
Show the output from ``pyvista report --help``.

.. command-output:: pyvista report --help

.. _cli_validate:

Validate
--------
The ``pyvista validate`` command is similar to calling :meth:`~pyvista.DataObjectFilters.validate_mesh`
in Python.

Examples
********
Validate a PLY mesh file.

.. command-output:: pyvista validate ant.ply
   :cwd: /_local_examples

Only validate its cells, and exclude the ``non_convex`` field.

.. command-output:: pyvista validate ant.ply --fields cells --exclude non_convex
   :cwd: /_local_examples

Use wildcard patterns to validate all mesh files in a directory.

.. command-output:: pyvista validate *.*
   :cwd: /_local_examples

API Reference
*************
Show the output from ``pyvista validate --help``.

.. command-output:: pyvista validate --help
