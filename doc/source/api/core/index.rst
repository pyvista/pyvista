.. _core-api-index:

Core API
========

PyVista wraps the `Visualization Toolkit`_ (VTK) mesh data types with a
Pythonic API. All PyVista meshes inherit from :class:`~pyvista.DataSet`
and provide direct access to common VTK filters (see :ref:`filters`).

.. _Visualization Toolkit: https://vtk.org

.. toctree::
   :hidden:

   objects
   dataset
   pointsets
   grids
   composite
   partitioned
   filters
   accessors
   camera
   lights
   cells
   helpers
   misc
   typing


Data Types
----------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Point Sets
      :link: point_sets_api
      :link-type: ref
      :class-title: pyvista-card-title

      Datasets with explicit point and cell geometry: surface meshes,
      volumetric meshes, and point clouds.

   .. grid-item-card:: Gridded Data
      :link: grids_api
      :link-type: ref
      :class-title: pyvista-card-title

      Datasets with topologically regular point spacing: uniform grids and
      rectilinear grids.

   .. grid-item-card:: Composite Datasets
      :link: composite_api
      :link-type: ref
      :class-title: pyvista-card-title

      Containers that hold multiple datasets in one object.

   .. grid-item-card:: Base Classes
      :link: dataset
      :link-type: ref
      :class-title: pyvista-card-title

      DataObject, DataSet, DataSetAttributes, Table, and Texture.

   .. grid-item-card:: Filters
      :link: filters
      :link-type: ref
      :class-title: pyvista-card-title

      All filtering methods available on PyVista datasets.

   .. grid-item-card:: Scene Objects
      :link: cameras_api
      :link-type: ref
      :class-title: pyvista-card-title

      Camera, Light, Cell, and CellArray.


Shared Base Classes
-------------------
These classes are not used directly. They are documented because they define
members shared by classes across the core and plotting APIs, so that each of
those members is documented once, here, and linked from every class that
inherits it.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   core.utilities.misc._BoundsSizeMixin
   core.utilities.misc._NameMixin
