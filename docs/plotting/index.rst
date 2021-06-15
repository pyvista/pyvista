Plotting
========

* Intuitive plotting routines with ``matplotlib`` similar syntax (see :ref:`plotting_ref`)
* Plotting tools built for interactivity (see :ref:`widgets`)

.. toctree::
   :maxdepth: 2

   plotting
   widgets
   qt_plotting


Plotting API Reference
----------------------
Plotting module API reference.  These plotting modules are the basis for
all plotting functionality in PyVista.

.. currentmodule:: pyvista

.. autosummary::
   :toctree:
   :template: custom-class-template.rst

   Plotter
   Renderer


Convenience Functions
~~~~~~~~~~~~~~~~~~~~~
These functions provide a simplified interface to the various plotting
routines in PyVista.

.. autosummary::
   :toctree: api-autogen/

   pyvista.plot
   pyvista.plot_arrows
   pyvista.set_plot_theme
   pyvista.create_axes_orientation_box
