.. _cli_convert:

pyvista convert
---------------
Command-line interface for converting one or more mesh files to another format.

Using ``pyvista convert`` is similar to calling :func:`~pyvista.read` and
:func:`~pyvista.DataObject.save` together in Python.

Examples
********

.. note::
    To run the examples yourself locally, first change directory to ``pyvista/examples``, e.g.

    .. code-block:: bash

        cd $(python -c "import pyvista.examples, pathlib; print(pathlib.Path(pyvista.examples.__file__).parent)")

Convert a PLY file (``.ply``) into a VTK XML PolyData file (``.vtp``).

.. command-output:: pyvista convert ant.ply ant_converted.vtp
   :cwd: /_local_examples

Keep the filename the same and specify the output extension only.

.. command-output:: pyvista convert ant.ply .vtp
   :cwd: /_local_examples

Use wildcard patterns to convert multiple files at once into an explicit output directory.

.. command-output:: pyvista convert *.ply output/.vtp
   :cwd: /_local_examples

API Reference
*************
Show the output from ``pyvista convert --help``.

.. command-output:: pyvista convert --help
