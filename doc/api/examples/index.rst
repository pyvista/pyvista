Examples
========
PyVista contains a variety of built-in demos and downloadable example
datasets.  For example:

.. pyvista-plot::

   Plot the built-in globe dataset

   >>> from pyvista import examples
   >>> globe = examples.load_globe()
   >>> globe.plot()

Many datasets are too large to be included with PyVista, but can be
downloaded and cached locally.  These datasets can be downloaded and
used with:

.. pyvista-plot::

   Plot the turbine blade mesh.

   >>> from pyvista import examples
   >>> blade_mesh = examples.download_turbine_blade()
   >>> blade_mesh.plot()

Finally, PyVista contains some demos which can be used to quickly
demonstrate features.

.. pyvista-plot::

   Create and show the orientation cube plotter

   >>> from pyvista import demos
   >>> plotter = demos.orientation_plotter()
   >>> plotter.show()


.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst

   examples.examples
   examples.downloads
   demos.demos
