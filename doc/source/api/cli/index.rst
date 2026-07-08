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

.. toctree::
   :hidden:

   plot
   convert
   validate
   report
