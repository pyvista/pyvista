Plotting
========

* Intuitive plotting routines with `matplotlib`_ like syntax (see :ref:`plotting_ref`).
* Plotting tools built for interactivity (see :ref:`widgets`).

.. toctree::
   :maxdepth: 2

   plotting
   qt_plotting
   theme

.. _matplotlib: https://matplotlib.org/


Plotting API Reference
----------------------
Plotting module API reference.  These plotting modules are the basis for
all plotting functionality in PyVista.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   Plotter
   Renderer


Charts API
----------
Charts API reference. These dedicated classes can be used to embed
charts in plotting windows. Note that using charts requires a VTK version
of at least 9.0 and in a future release will require `vtk>=9.2`.

.. toctree::
   :maxdepth: 2

   charts/index


Widget API
----------
The :class:`pyvista.Plotter` class inherits all of the widget methods in
:class:`pyvista.WidgetHelper`, so all of the following methods
are available from any PyVista plotter.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   WidgetHelper


Convenience Functions
~~~~~~~~~~~~~~~~~~~~~
These functions provide a simplified interface to the various plotting
routines in PyVista.

.. toctree::
   :maxdepth: 2

   conv_func

