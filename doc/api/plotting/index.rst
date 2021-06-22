Plotting
========

* Intuitive plotting routines with ``matplotlib`` similar syntax (see :ref:`plotting_ref`).
* Plotting tools built for interactivity (see :ref:`widgets`).

.. toctree::
   :maxdepth: 2

   plotting
   qt_plotting


Plotting API Reference
----------------------
Plotting module API reference.  These plotting modules are the basis for
all plotting functionality in PyVista.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   BasePlotter
   Plotter
   Renderer


Widget API
----------
The :class:`pyvista.BasePlotter` class inherits all of the widget methods in
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


