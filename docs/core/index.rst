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


All PyVista meshes inherit from the Common dataset type (see :ref:`ref_common`).
PyVista has the following mesh types:

- :class:`pyvista.PolyData` consists of any 1D or 2D geometries to construct vertices, lines, polygons, and triangles. We generally use ``PolyData`` to construct scattered points and closed/open surfaces (non-volumetric datasets). The :class:`pyvista.PolyData` class is an extension of ``vtk.vtkPolyData``.

- An :class:`pyvista.UnstructuredGrid` is the most general dataset type that can hold any 1D, 2D, or 3D cell geometries. You can think of this as a 3D extension of ``PolyData`` that allows volumetric cells to be present. It's fairly uncommon to explicitly make unstructured grids but they are often the result of different processing routines that might extract subsets of larger datasets. The :class:`pyvista.UnstructuredGrid` class is an extension of ``vtk.UnstructuredGrid``.

- A :class:`pyvista.StructuredGrid` is a regular lattice of points aligned with an internal coordinate axes such that the connectivity can be defined by a grid ordering. These are commonly made from :func:`np.meshgrid`. The cell types of structured grids must be 2D Quads or 3D Hexahedrons. The :class:`pyvista.StructuredGrid` class is an extension of ``vtk.vtkStructuredGrid``.

- A :class:`pyvista.RectilinearGrid` defines meshes with implicit geometries along the axes directions that are rectangular and regular. The :class:`pyvista.RectilinearGrid` class is an extension of ``vtk.vtkRectilinearGrid``.

- Image data, commonly referenced to as uniform grids, and defined by the :class:`pyvista.UniformGrid` class are meshes with implicit geometries where cell sizes are uniformly assigned along each axis and the spatial reference is built out from an origin point. The :class:`pyvista.UniformGrid` class is an extension of ``vtk.vtkImageData``.

- :class:`pyvista.MultiBlock` datasets are containers to hold several VTK datasets in one accessible and spatially referenced object. The :class:`pyvista.MultiBlock` class is an extension of ``vtk.vtkMultiBlockDataSet``.




.. toctree::
   :maxdepth: 2

   objects
   common
   points
   point-grids
   grids
   composite
   filters
