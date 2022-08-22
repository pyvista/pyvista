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

   examples.examples
   examples.downloads
   demos.demos

.. warning::
   As you browse this repository and think about how you might use our 3D models and range datasets, please remember that several of these artifacts have religious or cultural significance.
   Aside from the buddha, which is a religious symbol revered by hundreds of millions of people, the dragon is a symbol of Chinese culture, the Thai statue contains elements of religious significance to Hindus, and Lucy is a Christian angel; statues like her are commonly seen in Italian churches.
   Keep your renderings and other uses of these particular models in good taste.
   Don't animate or morph them, don't apply Boolean operators to them, and don't simulate nasty things happening to them (like breaking, exploding, melting, etc.).
   Choose another model for these sorts of experiments.
   (You can do anything you want to the Stanford bunny or the armadillo.)
