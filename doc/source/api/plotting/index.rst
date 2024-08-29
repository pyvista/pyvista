.. _plotting-api-index:

Plotting
========

* Intuitive plotting routines with `matplotlib`_ like syntax (see :ref:`plotting`).
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
Plotting module API reference. These plotting modules are the basis for
all plotting functionality in PyVista.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   Actor
   Axes
   AxesActor
   AxesAssembly
   AxesAssemblySymmetric
   CameraPosition
   CornerAnnotation
   CubeAxesActor
   DataSetMapper
   Label
   LookupTable
   PlanesAssembly
   Plotter
   Prop3D
   Property
   Renderer
   RenderWindowInteractor
   Text
   TextProperty
   Timer
   plotting.mapper._BaseMapper
   plotting.opts.InterpolationType
   plotting.opts.RepresentationType
   plotting.opts.ElementType
   plotting.volume.Volume
   plotting.volume_property.VolumeProperty


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
The :class:`pyvista.Plotter` class inherits all of the widget
methods described by the :class:`pyvista.plotting.widgets.WidgetHelper`
class. For additional details, see the
:ref:`widgets` examples.

.. autosummary::
   :toctree: _autosummary

   plotting.widgets.WidgetHelper
   plotting.widgets.AffineWidget3D


Picking API
-----------
The :class:`pyvista.Plotter` class inherits all of the picking
methods described by the :class:`pyvista.plotting.picking.PickingHelper`
class.

.. autosummary::
   :toctree: _autosummary

   plotting.picking.PickingHelper


Convenience Functions
~~~~~~~~~~~~~~~~~~~~~
These functions provide a simplified interface to the various plotting
routines in PyVista.

.. toctree::
   :maxdepth: 2

   conv_func
