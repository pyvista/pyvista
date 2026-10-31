"""Filters module with a class to manage filters/algorithms for unstructured grid datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.core.celltype import CellType
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.filters.poly_data import PolyDataFilters
from pyvista.core.utilities.misc import _wraps
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from pyvista import UnstructuredGrid
    from pyvista.core._typing_core._dataset_types import _UnstructuredGridType
    from pyvista.core.pointset import PolyData


# The general cell of each axis-aligned cell, and its corners in the general cell's order
_GENERAL_CELLS = {
    CellType.VOXEL: (CellType.HEXAHEDRON, [0, 1, 3, 2, 4, 5, 7, 6]),
    CellType.PIXEL: (CellType.QUAD, [0, 1, 3, 2]),
}


@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

    @_wraps(PolyDataFilters.delaunay_2d)
    def delaunay_2d(self, *args, **kwargs) -> PolyData:  # numpydoc ignore=PR01,RT01
        """Wrap ``PolyDataFilters.delaunay_2d``."""
        return PolyDataFilters.delaunay_2d(self, *args, **kwargs)  # type: ignore[arg-type]

    @_wraps(PolyDataFilters.reconstruct_surface)
    def reconstruct_surface(self, *args, **kwargs) -> PolyData:  # numpydoc ignore=PR01,RT01
        """Wrap ``PolyDataFilters.reconstruct_surface``."""
        return PolyDataFilters.reconstruct_surface(self, *args, **kwargs)  # type: ignore[arg-type]

    def subdivide_tetra(self) -> UnstructuredGrid:
        """Subdivide each tetrahedron into twelve tetrahedrons.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing the subdivided tetrahedrons.

        Examples
        --------
        First, load a sample tetrahedral UnstructuredGrid and plot it.

        >>> from pyvista import examples
        >>> grid = examples.load_tetbeam()
        >>> grid.plot(show_edges=True, line_width=2)

        Subdivide and plot the result.

        >>> subdivided = grid.subdivide_tetra()
        >>> subdivided.plot(show_edges=True, line_width=2)

        """
        alg = _vtk.vtkSubdivideTetra()
        alg.SetInputData(self)
        _update_alg(alg)
        return _get_output(alg, output_type=pv.UnstructuredGrid)

    def clean(
        self,
        *,
        tolerance: float = 0,
        remove_unused_points: bool = True,
        produce_merge_map: bool = True,
        average_point_data: bool = True,
        merging_array_name: str | None = None,
        progress_bar: bool = False,
    ) -> UnstructuredGrid:
        """Merge duplicate points and remove unused points in an UnstructuredGrid.

        This filter, merging coincident points as defined by a merging
        tolerance and optionally removes unused points. The filter does not
        modify the topology of the input dataset, nor change the types of
        cells. It may however, renumber the cell connectivity ids.

        This filter implements :vtk:`vtkStaticCleanUnstructuredGrid`.

        Parameters
        ----------
        tolerance : float, default: 0.0
            The absolute point merging tolerance.

        remove_unused_points : bool, default: True
            Indicate whether points unused by any cell are removed from the
            output. Note that when this is off, the filter can successfully
            process datasets with no cells (and just points). If on in this
            case, and there are no cells, the output will be empty.

        produce_merge_map : bool, default: False
            Indicate whether a merge map should be produced on output.
            The merge map, if requested, maps each input point to its
            output point id, or provides a value of -1 if the input point
            is not used in the output. The merge map is associated with
            the filter's output field data and is named ``"PointMergeMap"``.

        average_point_data : bool, default: True
            Indicate whether point coordinates and point data of merged points
            are averaged. When ``True``, the data coordinates and attribute
            values of all merged points are averaged. When ``False``, the point
            coordinate and data of the single remaining merged point is
            retained.

        merging_array_name : str, optional
            If a ``merging_array_name`` is specified and exists in the
            ``point_data``, then point merging will switch into a mode where
            merged points must be both geometrically coincident and have
            matching point data. When set, ``tolerance`` has no effect.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        UnstructuredGrid
            Cleaned unstructured grid.

        See Also
        --------
        remove_unused_points
            Strictly remove unused points `without` merging points.
        :meth:`~pyvista.DataObjectFilters.cell_quality`
        :meth:`~pyvista.DataObjectFilters.cell_validator`
        :meth:`~pyvista.DataObjectFilters.validate_mesh`

        Examples
        --------
        Demonstrate cleaning an UnstructuredGrid and show how it can be used to
        average the point data across merged points.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hexbeam = examples.load_hexbeam()
        >>> hexbeam_shifted = hexbeam.translate([1, 0, 0])
        >>> hexbeam.point_data['data'] = [0] * hexbeam.n_points
        >>> hexbeam_shifted.point_data['data'] = [1] * hexbeam.n_points
        >>> merged = hexbeam.merge(hexbeam_shifted, merge_points=False)
        >>> cleaned = merged.clean(average_point_data=True)
        >>> cleaned.n_points < merged.n_points
        True

        Show how point averaging using the ``clean`` method with
        ``average_point_data=True`` results in averaged point data for merged
        points.

        >>> pl = pv.Plotter(shape=(1, 2))
        >>> _ = pl.add_mesh(merged, scalars='data', show_scalar_bar=False)
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_mesh(cleaned, scalars='data')
        >>> pl.show()

        """
        alg = _vtk.vtkStaticCleanUnstructuredGrid()
        # https://github.com/pyvista/pyvista/pull/6337
        alg.SetInputDataObject(self.copy())  # type: ignore[attr-defined]
        alg.SetAbsoluteTolerance(True)
        alg.SetTolerance(tolerance)
        alg.SetMergingArray(merging_array_name)  # type: ignore[arg-type]
        alg.SetRemoveUnusedPoints(remove_unused_points)
        alg.SetProduceMergeMap(produce_merge_map)
        alg.SetAveragePointData(average_point_data)
        _update_alg(alg, progress_bar=progress_bar, message='Cleaning Unstructured Grid')
        return _get_output(alg, output_type=pv.UnstructuredGrid)

    def generalize_cells(  # type: ignore[misc]
        self: _UnstructuredGridType,
        *,
        inplace: bool = False,
    ) -> _UnstructuredGridType:
        """Convert voxels to hexahedra and pixels to quads.

        :attr:`~pyvista.CellType.VOXEL` and :attr:`~pyvista.CellType.PIXEL` cells must stay
        aligned with the axes. Their :attr:`~pyvista.CellType.HEXAHEDRON` and
        :attr:`~pyvista.CellType.QUAD` counterparts can take any orientation, for example
        after a rotation or a reflection across a plane that is not normal to an axis. The
        points and data are unchanged, and other cells are kept as they are.

        :meth:`~pyvista.DataSetFilters.voxelize` outputs voxels too, and converts them itself
        with ``generalize_cells=True``.

        .. versionadded:: 0.50

        Parameters
        ----------
        inplace : bool, default: False
            If ``True`` the mesh is updated in-place, otherwise a copy is returned.

        Returns
        -------
        UnstructuredGrid
            Mesh with hexahedra and quads in place of voxels and pixels.

        See Also
        --------
        pyvista.Grid.to_hexahedra
        pyvista.Grid.to_quads
        pyvista.DataSetFilters.voxelize
        pyvista.UnstructuredGrid.linear_copy

        Examples
        --------
        Convert the voxels of an image cast to an unstructured grid.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
        >>> grid.distinct_cell_types
        {<CellType.VOXEL: 11>}
        >>> grid.generalize_cells().distinct_cell_types
        {<CellType.HEXAHEDRON: 12>}

        The hexahedra stay valid after a reflection that would turn the voxels away from
        the axes.

        >>> reflected = grid.generalize_cells().reflect((1, 1, 1))
        >>> reflected.validate_mesh().is_valid
        True

        """
        output = self if inplace else self.copy(deep=False)
        celltypes = self.celltypes
        if not np.isin(celltypes, list(_GENERAL_CELLS)).any():
            return output

        offsets = self.cell_offsets
        connectivity = self.cell_connectivity.copy()
        general_types = celltypes.copy()
        for cell_type, (general_type, order) in _GENERAL_CELLS.items():
            ids = np.flatnonzero(celltypes == cell_type)
            rows = offsets[ids, np.newaxis] + np.arange(len(order))
            connectivity[rows] = connectivity[rows][:, order]
            general_types[ids] = general_type

        cells = pv.CellArray.from_arrays(offsets, connectivity)
        vtk_cell_types = _vtk.numpy_to_vtk(general_types, deep=True)
        if pv.vtk_version_info >= (9, 5):
            face_locations = self.GetPolyhedronFaceLocations()
            faces = self.GetPolyhedronFaces()
            output.SetPolyhedralCells(vtk_cell_types, cells, face_locations, faces)
        else:
            output.SetCells(vtk_cell_types, self.GetCellLocationsArray(), cells)
        return output

    def remove_unused_points(  # type: ignore[misc]
        self: _UnstructuredGridType,
        *,
        inplace: bool = False,
    ) -> _UnstructuredGridType:
        """Remove points which are not used by any cells.

        Unlike :meth:`clean`, this filter does `not` merge points.

        .. versionadded:: 0.46

        Parameters
        ----------
        inplace : bool, default: False
            If ``True`` the mesh is updated in-place, otherwise a copy is returned.

        See Also
        --------
        pyvista.PolyDataFilters.remove_unused_points

        Returns
        -------
        UnstructuredGrid
            Mesh with unused points removed.

        Examples
        --------
        Create :class:`~pyvista.UnstructuredGrid` with three points. The first two points are
        coincident and associated with :attr:`~pyvista.CellType.VERTEX` cells, and the third point
        is "unused" and not associated with any cells.

        >>> import pyvista as pv
        >>> cells = [1, 0, 1, 1]
        >>> celltypes = [pv.CellType.VERTEX, pv.CellType.VERTEX]
        >>> points = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        >>> grid = pv.UnstructuredGrid(cells, celltypes, points)
        >>> grid
        UnstructuredGrid (...)
          N Cells:    2
          N Points:   3
          X Bounds:   0.000e+00, 1.000e+00
          Y Bounds:   0.000e+00, 1.000e+00
          Z Bounds:   0.000e+00, 1.000e+00
          N Arrays:   0

        Since the third point is unused, we can remove it. Note that coincident points are `not`
        merged by this filter, so the two vertex points are kept as-is.

        >>> grid = grid.remove_unused_points()
        >>> grid
        UnstructuredGrid (...)
          N Cells:    2
          N Points:   2
          X Bounds:   0.000e+00, 0.000e+00
          Y Bounds:   0.000e+00, 0.000e+00
          Z Bounds:   0.000e+00, 0.000e+00
          N Arrays:   0

        """
        if self.is_empty:
            return self if inplace else self.copy()

        out = self if inplace else self.copy()
        cell_ids = np.arange(out.n_cells)

        # Need to add an extra "dummy" cell to force vtkExtractCells to remap the point IDs
        cell_array = out.GetCells()
        if inplace:
            # Another mesh may share the input's cell array with it
            grown = _vtk.vtkCellArray()
            grown.DeepCopy(cell_array)
            out.SetCells(out._get_cell_types_array(), grown)
            cell_array = grown
        cell_array.InsertNextCell(1)

        # Extract all the cells, except for the dummy cell
        extracted = out.extract_cells(cell_ids, pass_point_ids=False, pass_cell_ids=False)

        out.copy_from(extracted, deep=False)
        return out
