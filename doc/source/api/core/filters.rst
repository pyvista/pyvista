.. _filters:

Filters
-------

.. currentmodule:: pyvista

.. _points_dtype_support:

.. note::

   A filter's output :attr:`~pyvista.DataSet.points` dtype is decided by
   :attr:`pyvista.core.config.Config.points_dtype`, which is ``None`` by default and
   leaves each algorithm to produce whatever it produces. Not every VTK algorithm can
   generate double-precision points, so each section below marks which of its filters
   can:

   .. list-table::
      :widths: 8 92

      * - :material-regular:`check;2em;sd-text-success`
        - Generates double-precision points when asked for them.
      * - :material-regular:`remove;2em;sd-text-warning`
        - Depends on the type of mesh passed in, because the filter chooses its
          algorithm from it. :meth:`~pyvista.DataSetFilters.contour` delivers double
          precision from a :class:`~pyvista.PolyData` and cannot from an
          :class:`~pyvista.ImageData`.
      * - :material-regular:`close;2em;sd-text-error`
        - Cannot. PyVista casts the single-precision output up so the dtype is the one
          requested, and raises :class:`~pyvista.PrecisionWarning` to say the
          values behind it are not.

   These marks are measured by running each filter while the documentation is built, so
   they describe the VTK release PyVista is built against here. Another release, or a
   different backend such as ``cvista``, may draw the line elsewhere. To settle it for
   the build you are actually using, set ``points_dtype = 'float64'`` and run your own
   pipeline: any algorithm that cannot deliver it names itself in the warning.

Data Object Filters
~~~~~~~~~~~~~~~~~~~
The :class:`pyvista.DataObjectFilters` is inherited by :class:`pyvista.DataSet`
and :class:`pyvista.MultiBlock`. These filters are the most general and are
available as callable methods directly from any PyVista dataset or multi-block.

.. autosummary::
   :toctree: _autosummary

   DataObjectFilters

.. include:: /api/core/points_dtype/data_object_filters.rst

Dataset Filters
~~~~~~~~~~~~~~~
The :class:`pyvista.DataSetFilters` is inherited by :class:`pyvista.DataSet` making
all the following filters available as callable methods directly from any
PyVista dataset.

.. autosummary::
   :toctree: _autosummary

   DataSetFilters

.. include:: /api/core/points_dtype/data_set_filters.rst


PolyData Filters
~~~~~~~~~~~~~~~~
The :class:`pyvista.PolyDataFilters` is inherited by :class:`pyvista.PolyData`
making all the following filters available as callable methods directly
from any ``PolyData`` mesh.

.. autosummary::
   :toctree: _autosummary

   PolyDataFilters

.. include:: /api/core/points_dtype/poly_data_filters.rst


UnstructuredGrid Filters
~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`pyvista.UnstructuredGridFilters` is inherited by
:class:`pyvista.UnstructuredGrid` making all the following filters
available as callable methods directly from any ``UnstructuredGrid`` mesh.

.. autosummary::
   :toctree: _autosummary

   UnstructuredGridFilters

.. include:: /api/core/points_dtype/unstructured_grid_filters.rst


ImageData Filters
~~~~~~~~~~~~~~~~~
The :class:`pyvista.ImageDataFilters` is inherited by
:class:`pyvista.ImageData` making all the following filters
available as callable methods directly from any ``ImageData`` mesh.

.. autosummary::
   :toctree: _autosummary

   ImageDataFilters

.. include:: /api/core/points_dtype/image_data_filters.rst


Composite Filters
~~~~~~~~~~~~~~~~~
These are filters that can be applied to composite datasets, that is
:class:`pyvista.MultiBlock`. The :class:`pyvista.CompositeFilters` class
inherits many but not all of the filters from :class:`pyvista.DataSetFilters`.

.. autosummary::
   :toctree: _autosummary

   CompositeFilters

.. include:: /api/core/points_dtype/composite_filters.rst
