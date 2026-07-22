.. _cli_api:

Command Line
============

PyVista ships a command-line interface (CLI) for :ref:`plotting <cli_plot>`,
:ref:`converting <cli_convert>`, and :ref:`validating <cli_validate>`
mesh files, as well as generating environment :ref:`reports <cli_report>`,
without requiring Python code.
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

Show the output from ``pyvista --help`` to show all available arguments and subcommands.

.. command-output:: pyvista --help

The API for each subcommand is documented below.

.. toctree::
   :caption: Subcommands
   :maxdepth: 1

   plot
   convert
   validate
   report
