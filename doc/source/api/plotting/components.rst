.. _plotter-component-api:

Plotter Components
==================

PyVista exposes a namespaced *component* extension point on
:class:`pyvista.BasePlotter` so that third-party packages can attach
namespaced helpers to the plotter without monkey-patching or
subclassing. Once a plugin module is imported, its component becomes
available on every plotter instance as
``plotter.<namespace>.<method>(...)``.

Components are constructed lazily on first attribute access and cached
on the plotter instance. They participate in the plotter lifecycle: a
component class can define optional ``__plotter_close__`` and
``__plotter_deep_clean__`` dunder methods that PyVista invokes from
:meth:`pyvista.BasePlotter.close` and
:meth:`pyvista.BasePlotter.deep_clean`. Both hooks fire only on
components that were actually constructed (touched at least once), in
reverse construction order.

The registration surface mirrors the dataset-accessor API line for
line. Plugin authors learn one decorator pattern and apply it to both
datasets and the plotter.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   register_plotter_component
   unregister_plotter_component
   registered_plotter_components
   ComponentRegistration
   PlotterComponent
