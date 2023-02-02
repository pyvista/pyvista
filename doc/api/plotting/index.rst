.. _plotting-api-index:

Plotting
========

* Intuitive plotting routines with `matplotlib`_ like syntax (see :ref:`plotting_ref`).
* Plotting tools built for interactivity (see :ref:`widgets`).

.. toctree::
   :maxdepth: 2

   plotting
   qt_plotting
   theme
   trame

.. _matplotlib: https://matplotlib.org/


Plotting API Reference
----------------------
Plotting module API reference.  These plotting modules are the basis for
all plotting functionality in PyVista.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   BasePlotter
   Plotter
   Renderer
   Property
   plotting.volume_property.VolumeProperty
   plotting.volume.Volume
   Actor
   DataSetMapper
   LookupTable


Composite Plotting Reference
----------------------------
These classes are used when plotting :class:`pyvista.MultiBlock` datasets.

.. autosummary::
   :toctree: _autosummary

   CompositePolyDataMapper
   CompositeAttributes
   BlockAttributes


Charts API
----------
Charts API reference. These dedicated classes can be used to embed
charts in plotting windows. Note that full charts functionality
requires a VTK version of at least 9.3. Most components work fine
in older versions though.

.. toctree::
   :maxdepth: 2

   charts/index


Widget API
----------
The :class:`pyvista.Plotter` class inherits all of the widget methods described
by the ``pyvista.WidgetHelper`` class. For additional details, see the
:ref:`widgets` examples.


Convenience Functions
~~~~~~~~~~~~~~~~~~~~~
These functions provide a simplified interface to the various plotting
routines in PyVista.

.. toctree::
   :maxdepth: 2

   conv_func
