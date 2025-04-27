.. _filters:

Filters
-------

.. currentmodule:: pyvista

Data Object Filters
~~~~~~~~~~~~~~~~~~~
The :class:`pyvista.DataObjectFilters` is inherited by :class:`pyvista.DataSet`
and :class:`pyvista.MultiBlock`. These filters are the most general and are
available as callable methods directly from any PyVista dataset or multi-block.

.. autosummary::
   :toctree: _autosummary

   DataObjectFilters

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


ImageData Filters
~~~~~~~~~~~~~~~~~
The :class:`pyvista.ImageDataFilters` is inherited by
:class:`pyvista.ImageData` making all the following filters
available as callable methods directly from any ``ImageData`` mesh.

.. autosummary::
   :toctree: _autosummary

   ImageDataFilters


Composite Filters
~~~~~~~~~~~~~~~~~
These are filters that can be applied to composite datasets, that is
:class:`pyvista.MultiBlock`. The :class:`pyvista.CompositeFilters` class
inherits many but not all of the filters from :class:`pyvista.DataSetFilters`.

.. autosummary::
   :toctree: _autosummary

   CompositeFilters
