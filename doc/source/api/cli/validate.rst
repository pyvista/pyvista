.. _cli_validate:

pyvista validate
----------------
Command-line interface for validating one or more mesh files.

Using ``pyvista validate`` is similar to calling
:meth:`~pyvista.DataObjectFilters.validate_mesh` in Python.

Examples
********
.. note::
    To run the examples yourself locally, first change directory to ``pyvista/examples``, e.g.

    .. code-block:: bash

        cd $(python -c "import pyvista.examples, pathlib; print(pathlib.Path(pyvista.examples.__file__).parent)")

Validate a PLY mesh file.

.. command-output:: pyvista validate ant.ply
   :cwd: /_local_examples

Only validate its cells, and exclude the ``non_convex`` field.

.. command-output:: pyvista validate ant.ply --fields cells --exclude non_convex
   :cwd: /_local_examples

Use wildcard patterns to validate all mesh files in a directory.
Use ``--skip-unreadable`` to skip non-mesh files.

.. command-output:: pyvista validate *.* --skip-unreadable
   :cwd: /_local_examples

API Reference
*************
Show the output from ``pyvista validate --help``.

.. command-output:: pyvista validate --help
