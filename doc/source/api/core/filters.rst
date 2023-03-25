.. _filters_ref:

Filters
-------

.. currentmodule:: pyvista

Dataset Filters
~~~~~~~~~~~~~~~
The :class:`pyvista.DataSetFilters` is inherited by :class:`pyvista.DataSet` making
all the following filters available as callable methods directly from any
PyVista dataset.

.. autosummary::
   :toctree: _autosummary

   DataSetFilters


PolyData Filters
~~~~~~~~~~~~~~~~
The :class:`pyvista.PolyDataFilters` is inherited by :class:`pyvista.PolyData`
making all the following filters available as callable methods directly
from any ``PolyData`` mesh.

.. autosummary::
   :toctree: _autosummary

   PolyDataFilters


UnstructuredGrid Filters
~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`pyvista.UnstructuredGridFilters` is inherited by
:class:`pyvista.UnstructuredGrid` making all the following filters
available as callable methods directly from any ``UnstructuredGrid`` mesh.

.. autosummary::
   :toctree: _autosummary

   UnstructuredGridFilters


UniformGrid Filters
~~~~~~~~~~~~~~~~~~~
The :class:`pyvista.UniformGridFilters` is inherited by
:class:`pyvista.UniformGrid` making all the following filters
available as callable methods directly from any ``UniformGrid`` mesh.

.. autosummary::
   :toctree: _autosummary

   UniformGridFilters


Composite Filters
~~~~~~~~~~~~~~~~~
These are filters that can be applied to composite datasets, that is
:class:`pyvista.MultiBlock`. The :class:`pyvista.CompositeFilters` class
inherits many but not all of the filters from :class:`pyvista.DataSetFilters`.

.. autosummary::
   :toctree: _autosummary

   CompositeFilters
