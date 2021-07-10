.. _ref_plot_directive_docs:

Sphinx PyVista Plot Directive
=============================
You can generate static images of pyvista plots using the
``.. pyvista-plot`` directive by adding the following to your
``conf.py`` when building your documentation using Sphinx.

.. code:: python

   extensions = [
       "sphinx.ext.napoleon",
       "pyvista.ext.plot_directive",
   ]

You can then issue the plotting directive within your sphinx
documentation files:

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


.. automodule::
   pyvista.ext.plot_directive
