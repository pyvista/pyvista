.. _ref_dataset:

Datasets
========

Datasets are any spatially referenced information and usually consist of
geometrical representations of a surface or volume in 3D space.
In VTK, this superclass is represented by the ``vtk.vtkDataSet`` abstract class.

In VTK, datasets consist of geometry, topology, and attributes to which PyVista
provides direct access:

* Geometry is the collection of points and cells in 2D or 3D space.
* Topology defines the structure of the dataset, or how the points are connected to each other to form a cells making a surface or volume.
* Attributes are any data values that are associated to either the points or cells of the dataset

All of the following data types are listed subclasses of a dataset and share a
set of common functionality which we wrap into the base class
:class:`pyvista.DataSet`.


The DataSet Model
-----------------

The :class:`pyvista.DataSet` class holds attributes that are *common* to all
spatially referenced datasets in PyVista.
This base class is analogous to VTK's ``vtk.vtkDataSet`` class.


.. rubric:: Attributes

.. autoautosummary:: pyvista.DataSet
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.DataSet
   :methods:


.. autoclass:: pyvista.DataSet
   :show-inheritance:
   :members:
   :undoc-members:
