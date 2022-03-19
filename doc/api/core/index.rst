Core API
========

The `Visualization Toolkit`_ (VTK), developed by Kitware_, has many mesh data
types that PyVista wraps.
This chapter is intended to describe these different mesh types and how we wrap
and implement each of those mesh types in VTK. This chapter also highlights
how all PyVista types have direct access to common VTK filters
(see :ref:`filters_ref`).

.. _Visualization Toolkit: https://vtk.org
.. _Kitware: https://www.kitware.com


All PyVista meshes inherit from the DataSet type (see :ref:`ref_dataset`).
PyVista has the following mesh types:

- :class:`pyvista.PolyData` consists of any 1D or 2D geometries to construct vertices, lines, polygons, and triangles. We generally use :class:`pyvista.PolyData` to construct scattered points and closed/open surfaces (non-volumetric datasets). The :class:`pyvista.PolyData` class is an extension of `vtk.vtkPolyData`_.

- A :class:`pyvista.UnstructuredGrid` is the most general dataset type that can hold any 1D, 2D, or 3D cell geometries. You can think of this as a 3D extension of :class:`pyvista.PolyData` that allows volumetric cells to be present. It's fairly uncommon to explicitly make unstructured grids but they are often the result of different processing routines that might extract subsets of larger datasets. The :class:`pyvista.UnstructuredGrid` class is an extension of `vtk.vtkUnstructuredGrid`_.

- A :class:`pyvista.StructuredGrid` is a regular lattice of points aligned with internal coordinate axes such that the connectivity can be defined by a grid ordering. These are commonly made from :func:`numpy.meshgrid`. The cell types of structured grids must be 2D quads or 3D hexahedra. The :class:`pyvista.StructuredGrid` class is an extension of `vtk.vtkStructuredGrid`_.

- A :class:`pyvista.RectilinearGrid` defines meshes with implicit geometries along the axis directions that are rectangular and regular. The :class:`pyvista.RectilinearGrid` class is an extension of `vtk.vtkRectilinearGrid`_.

- Image data, commonly referred to as uniform grids, and defined by the :class:`pyvista.UniformGrid` class are meshes with implicit geometries where cell sizes are uniformly assigned along each axis and the spatial reference is built out from an origin point. The :class:`pyvista.UniformGrid` class is an extension of `vtk.vtkImageData`_.

- :class:`pyvista.MultiBlock` datasets are containers to hold several VTK datasets in one accessible and spatially referenced object. The :class:`pyvista.MultiBlock` class is an extension of `vtk.vtkMultiBlockDataSet`_.

.. _vtk.vtkPolyData: https://vtk.org/doc/nightly/html/classvtkPolyData.html
.. _vtk.vtkUnstructuredGrid: https://vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html
.. _vtk.vtkStructuredGrid: https://vtk.org/doc/nightly/html/classvtkStructuredGrid.html
.. _vtk.vtkRectilinearGrid: https://vtk.org/doc/nightly/html/classvtkRectilinearGrid.html
.. _vtk.vtkImageData: https://vtk.org/doc/nightly/html/classvtkImageData.html
.. _vtk.vtkMultiBlockDataSet: https://vtk.org/doc/nightly/html/classvtkMultiBlockDataSet.html

.. toctree::
   :maxdepth: 2

   objects
   dataset
   pointsets
   grids
   composite
   filters
   camera
   lights
   helpers
