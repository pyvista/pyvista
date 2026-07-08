.. _cli_api:

Command Line Interface
======================

PyVista ships a command-line interface for plotting, converting, and validating
one or more mesh files, and for generating environment reports, without
needing to write any Python. The CLI is included with a typical PyVista
installation, e.g.:

.. code-block:: bash

    pip install pyvista

.. command-output:: pyvista --help

Each subcommand is documented below. The output shown is generated directly
from ``--help``.

.. _cli_convert:

Convert
-------

.. command-output:: pyvista convert --help
   :env: NO_COLOR=1

.. versionadded:: 0.47

.. _cli_plot:

Plot
----

.. command-output:: pyvista plot --help
   :env: NO_COLOR=1

.. versionadded:: 0.47

.. _cli_report:

Report
------

.. command-output:: pyvista report --help
   :env: NO_COLOR=1

.. versionadded:: 0.47

.. _cli_validate:

Validate
--------

.. command-output:: pyvista validate --help
   :env: NO_COLOR=1

.. versionadded:: 0.48
