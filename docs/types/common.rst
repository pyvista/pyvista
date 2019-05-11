Datasets
========

Datasets are any spatially reference information and usually consist of
geometrical representations of a surface or volume in 3D space.
In VTK, this superclass is represented by the ``vtk.vtkDataSet`` abstract class.

In VTK, datasets consist of geometry, topology, and attributes to which PyVista
provides direct access:

* Geometry is the collection of points and cells in 2D or 3D space.
* Topology defines the structure of the dataset, or how the points are connected to each other to form a cells making a surface or volume.
* Attributes are any data values that are associated to either the points or cells of the dataset

All of the following data types are listed subclasses of a dataset and share a
set of common functionality which we wrap into the base class
:class:`pyvista.Common`.



PolyData
--------

:class:`pyvista.PolyData` consists of any 1D or 2D geometries to construct
vertices, lines, polygons, and triangles.
We generally use ``PolyData`` to construct scattered points and closed/open
surfaces (non-volumetric datasets).
The :class:`pyvista.PolyData` class is an extension of ``vtk.vtkPolyData``.


UnstructuredGrid
----------------

An :class:`pyvista.UnstructuredGrid` is the most general dataset type that can hold
any 1D, 2D, or 3D cell geometries.
You can think of this as a 3D extension of ``PolyData`` that allows volumetric
cells to be present.
It's fairly uncommon to explicitly make unstructured grids but they are often
the result of different processing routines that might extract subsets of larger
datasets.
The :class:`pyvista.UnstructuredGrid` class is an extension of
``vtk.UnstructuredGrid``.


StructuredGrid
--------------

A :class:`pyvista.StructuredGrid` is a regular lattice of points aligned with an
internal coordinate axes such that the connectivity can be defined by a grid
ordering.
These are commonly made from :func:`np.meshgrid`. The cell types of structured
grids must be 2D Quads or 3D Hexahedrons.
The :class:`pyvista.StructuredGrid` class is an extension of
``vtk.vtkStructuredGrid``.


RectilinearGrid
---------------

A :class:`pyvista.RectilinearGrid` defines meshes with implicit geometries along the axes
directions that are rectangular and regular.
The :class:`pyvista.RectilinearGrid` class is an extension of
``vtk.vtkRectilinearGrid``.


ImageData
---------

Image data, commonly referenced to as uniform grids, and defined by the
:class:`pyvista.UniformGrid` class are meshes with implicit geometries where cell
sizes are uniformly assigned along each axis and the spatial reference is built
out from an origin point.
The :class:`pyvista.UniformGrid` class is an extension of ``vtk.vtkImageData``.


MultiBlock
----------

:class:`pyvista.MultiBlock` datasets are containers to hold several VTK datasets in
one accessible and spatially referenced object.
The :class:`pyvista.MultiBlock` class is an extension of
``vtk.vtkMultiBlockDataSet``.


The Common Model
================

The :class:`pyvista.Common` class holds attributes that are *common* to all
spatially referenced datasets in PyVista.
This base class is analogous to VTK's ``vtk.vtkDataSet`` class.


.. rubric:: Attributes

.. autoautosummary:: pyvista.Common
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.Common
   :methods:


.. autoclass:: pyvista.Common
   :show-inheritance:
   :members:
   :undoc-members:
