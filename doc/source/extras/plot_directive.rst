.. _plot_directive_docs:

Sphinx PyVista Plot Directive
=============================
You can generate static and interactive scenes of pyvista plots using the
``.. pyvista-plot::`` directive by adding the following to your
``conf.py`` when building your documentation using Sphinx.

.. code:: python

   extensions = [
       "pyvista.ext.plot_directive",
       "pyvista.ext.viewer_directive",
       "sphinx_design",
   ]

You can then issue the plotting directive within your sphinx
documentation files::

   .. pyvista-plot::
      :caption: A sphere
      :include-source: True

      >>> import pyvista
      >>> sphere = pyvista.Sphere()
      >>> out = sphere.plot()

Which will be rendered as:

.. pyvista-plot::
   :caption: This is a default sphere
   :include-source: True

   >>> import pyvista
   >>> sphere = pyvista.Sphere()
   >>> out = sphere.plot()

.. note::

   You need to install `the packages to build the interactive scene <https://github.com/pyvista/pyvista-docs-dynamic/blob/main/requirements.txt>`_.

.. note::

   You need to spin up a local server to view the interactive scene in the documentation.

   .. code-block:: bash

      python -m http.server 11000 --directory _build/html

.. automodule::
   pyvista.ext.plot_directive
