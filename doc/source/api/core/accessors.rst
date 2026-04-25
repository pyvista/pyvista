.. _accessor-api:

Dataset Accessors
=================

PyVista exposes a pandas/xarray-style accessor mechanism so that
third-party packages can add namespaced filter methods to dataset
classes without monkey-patching or subclassing. Once a plugin module
is imported, its accessor becomes available on every instance of the
target class as ``dataset.<namespace>.<method>(...)`` and chains
seamlessly with built-in PyVista filters.

See :ref:`extending-pyvista` for a full guide with a worked example
and guidance on typing and autocomplete.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   register_dataset_accessor
   unregister_dataset_accessor
   registered_accessors
   AccessorRegistration
   DataSetAccessor
