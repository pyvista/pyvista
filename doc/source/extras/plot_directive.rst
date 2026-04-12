.. _plot_directive_docs:

Sphinx PyVista Plot Directive
=============================
You can generate static and interactive scenes of pyvista plots using the
``.. pyvista-plot::`` directive by adding the following to your
``conf.py`` when building your documentation using Sphinx.

.. code-block:: python

    extensions = [
        "pyvista.ext.plot_directive",
        "pyvista.ext.viewer_directive",
        "sphinx_design",
    ]

You can then issue the plotting directive within your sphinx
documentation files::

   .. pyvista-plot::
      :caption: This is a default sphere
      :include-source: True

      >>> import pyvista as pv
      >>> sphere = pv.Sphere()
      >>> out = sphere.plot()

Which will be rendered as:

.. pyvista-plot::
   :caption: This is a default sphere
   :include-source: True

   >>> import pyvista as pv
   >>> sphere = pv.Sphere()
   >>> out = sphere.plot()

.. note::

   You need to install the following packages to build the interactive scene:

   .. code-block:: bash

      pip install 'pyvista[jupyter]' sphinx sphinx_design

.. note::

   You need to spin up a local server to view the interactive scene in the documentation.

   .. code-block:: bash

      python -m http.server 11000 --directory _build/html

Complete Example
================

The following is a script to build documentation with interactive plots
from scratch. The script will:

#. Create a new virtual environment and install dependencies
#. Create the files required for a simple documentation build:

   #. Sphinx configuration file ``doc/src/conf.py`` with extensions
   #. Source file ``doc/src/example.py`` with a simple plot directive example
   #. Index file ``doc/src/index.rst`` for site navigation

#. Build the documentation
#. Start a local server

You can copy and paste the script directly into a terminal and execute it.
Once the documentation is built, you should be able to view it with a web
browser by navigating to ``http://localhost:11000``.

.. code-block:: bash

    # Setup a new virtual environment and activate it
    python -m venv .venv
    emulate bash -c '. .venv/bin/activate'

    # Install dependencies for the build
    pip install 'pyvista[jupyter]' sphinx sphinx_design

    # Create new `doc/src` directory
    mkdir doc
    cd doc
    mkdir src

    # Create a simple python module and include an example
    # in the docstring using the plot directive.
    cat > src/example.py <<EOF

    def foo():
        """Some function.

        .. pyvista-plot::

            >>> import pyvista as pv
            >>> mesh = pv.Sphere()
            >>> mesh.plot()
        """

    EOF

    # Create the configuration file with the required extensions.
    # Here we also include `autodoc` for the example.
    cat > src/conf.py <<EOF
    import os, sys

    sys.path.insert(0, os.path.abspath("."))

    extensions = [
        "sphinx.ext.autodoc",
        "pyvista.ext.plot_directive",
        "pyvista.ext.viewer_directive",
        "sphinx_design",
    ]
    EOF

    # Create the index for the documentation
    cat > src/index.rst <<EOF
    API Reference
    =============

    .. automodule:: example
        :members:
        :undoc-members:
    EOF

    # Build the documentation
    sphinx-build -b html src _build/html

    # Start a local server for the interactive scene
    python -m http.server 11000 --directory _build/html


API Reference
=============

.. automodule::
   pyvista.ext.plot_directive
