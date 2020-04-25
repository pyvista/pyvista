.. _filters_ref:

Filters
-------


.. automodule:: pyvista.core.filters



Dataset Filters
~~~~~~~~~~~~~~~

The :class:`pyvista.DataSetFilters` is inherited by :class:`pyvista.Common` making
all the following filters available as callable methods directly from any
PyVista dataset.


.. rubric:: Methods

.. autoautosummary:: pyvista.DataSetFilters
   :methods:


.. autoclass:: pyvista.DataSetFilters
   :members:
   :undoc-members:



PolyData Filters
~~~~~~~~~~~~~~~~

The :class:`pyvista.PolyDataFilters` is inherited by :class:`pyvista.PolyData`
making all the following filters available as callable methods directly
from any ``PolyData`` mesh.


.. rubric:: Methods

.. autoautosummary:: pyvista.PolyDataFilters
   :methods:

.. autoclass:: pyvista.PolyDataFilters
   :show-inheritance:
   :members:
   :undoc-members:



UnstructuredGrid Filters
~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`pyvista.UnstructuredGridFilters` is inherited by
:class:`pyvista.UnstructuredGrid` making all the following filters
available as callable methods directly from any ``UnstructuredGrid`` mesh.

.. rubric:: Methods

.. autoautosummary:: pyvista.UnstructuredGridFilters
   :methods:

.. autoclass:: pyvista.UnstructuredGridFilters
   :show-inheritance:
   :members:
   :undoc-members:




UniformGrid Filters
~~~~~~~~~~~~~~~~~~~

The :class:`pyvista.UniformGridFilters` is inherited by
:class:`pyvista.UniformGrid` making all the following filters
available as callable methods directly from any ``UniformGrid`` mesh.

.. rubric:: Methods

.. autoautosummary:: pyvista.UniformGridFilters
   :methods:

.. autoclass:: pyvista.UniformGridFilters
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
