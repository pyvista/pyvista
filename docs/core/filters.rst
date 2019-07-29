.. _filters_ref:

Filters
-------


.. automodule:: pyvista.filters



Dataset Filters
~~~~~~~~~~~~~~~

The :class:`pyvista.DataSetFilters` is inherited by :class:`pyvista.Common` making
all the the following filters available as callable methods directly from any
PyVista dataset.


.. rubric:: Methods

.. autoautosummary:: pyvista.DataSetFilters
   :methods:


.. autoclass:: pyvista.DataSetFilters
   :members:
   :undoc-members:



PolyData Filters
~~~~~~~~~~~~~~~~


.. rubric:: Methods

.. autoautosummary:: pyvista.PolyDataFilters
   :methods:

.. autoclass:: pyvista.PolyDataFilters
   :show-inheritance:
   :members:
   :undoc-members:



UnstructuredGrid Filters
~~~~~~~~~~~~~~~~~~~~~~~~


.. rubric:: Methods

.. autoautosummary:: pyvista.UnstructuredGridFilters
   :methods:

.. autoclass:: pyvista.UnstructuredGridFilters
   :show-inheritance:
   :members:
   :undoc-members:



Composite Filters
~~~~~~~~~~~~~~~~~

These are filters that can be applied to composite datasets, i.e.
:class:`pyvista.MultiBlock`. The :class:`pyvista.CompositeFilters` class
inherits many but not all of the filters from :class:`pyvista.DataSetFilters`.

.. rubric:: Methods

.. autoautosummary:: pyvista.CompositeFilters
   :methods:

.. autoclass:: pyvista.CompositeFilters
   :show-inheritance:
   :members:
   :undoc-members:
