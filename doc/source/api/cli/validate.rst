.. _cli_validate:

pyvista validate
----------------
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
