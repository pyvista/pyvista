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


Open Graph link previews
========================

When a documentation link is shared, the preview that appears alongside it comes
from the page's `Open Graph <https://ogp.me>`_ metadata. PyVista fills that in
automatically for every page: the image is the plot the page renders.

This needs no configuration beyond enabling
`sphinxext-opengraph <https://github.com/wpilibsuite/sphinxext-opengraph>`_ and
telling it where the documentation is published:

.. code-block:: python

    extensions = [
        "pyvista.ext.plot_directive",
        "sphinxext.opengraph",
    ]

    ogp_site_url = "https://docs.example.org/"

``pyvista_plot_opengraph`` follows ``sphinxext.opengraph``: it defaults to ``None``,
which means "on if ``sphinxext.opengraph`` is enabled". Set it to ``True`` to require
it -- the build then fails if ``sphinxext.opengraph`` is missing -- or to ``False`` to
opt out.

Choosing the preview image
--------------------------

By default a page previews the first image it renders. Pages where the first image
is only scene-setting can select another one with the ``pyvista-plot-thumbnail``
directive:

.. code-block:: rst

   .. pyvista-plot-thumbnail:: 2

The argument is the one-based position of the image among *all* the page's plot
images, in the order they appear. It counts images, not files, so it is unaffected
by how the generated filenames happen to be numbered. Negative values count
backwards from the last image.

The directive renders nothing and can go anywhere on the page, which means it can
sit next to the code it refers to instead of at the top. In a docstring the natural
place is the start of the ``Examples`` section:

.. code-block:: rst

   Examples
   --------
   .. pyvista-plot-thumbnail:: 2

   Create a sphere.

   >>> import pyvista as pv
   >>> pv.Sphere().plot()

   Clip it, which is what this page is really about.

   >>> pv.Sphere().clip().plot()

A page has a single ``<head>``, so it has a single link preview -- section anchors
cannot have their own. Using the directive twice on one page therefore warns and
keeps the first selection. This can happen without either docstring being wrong, on
pages that document several objects at once with ``:members:``.

Selecting an image the page does not render also warns, and falls back to the first
image. That is silent while ``pyvista_plot_skip`` or ``pyvista_plot_skip_optional``
is enabled, since those builds deliberately render fewer images.

Sphinx-Gallery examples
-----------------------

Gallery examples already have a thumbnail, and their preview always matches it, so
that a shared link shows the same picture as the gallery. PyVista uses the full
resolution version of that image rather than the gallery's own thumbnail file, which
is too small to preview well.

Using ``pyvista-plot-thumbnail`` in a gallery example is an error. Use
Sphinx-Gallery's own selector instead:

.. code-block:: python

   # sphinx_gallery_thumbnail_number = 2

API Reference
=============

.. automodule::
   pyvista.ext.plot_directive
