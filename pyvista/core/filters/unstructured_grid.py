"""Filters module with a class to manage filters/algorithms for unstructured grid datasets."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import numpy as np

from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk
from pyvista.core.cell import CellArray
from pyvista.core.celltype import CellType
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.filters.poly_data import PolyDataFilters
from pyvista.core.utilities.misc import abstract_class

if TYPE_CHECKING:
    from pyvista import UnstructuredGrid


@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

    @wraps(PolyDataFilters.delaunay_2d)  # type: ignore[has-type]
    def delaunay_2d(self, *args, **kwargs):  # numpydoc ignore=PR01,RT01
        """Wrap ``PolyDataFilters.delaunay_2d``."""
        return PolyDataFilters.delaunay_2d(self, *args, **kwargs)  # type: ignore[arg-type]

    @wraps(PolyDataFilters.reconstruct_surface)  # type: ignore[has-type]
    def reconstruct_surface(self, *args, **kwargs):  # numpydoc ignore=PR01,RT01
        """Wrap ``PolyDataFilters.reconstruct_surface``."""
        return PolyDataFilters.reconstruct_surface(self, *args, **kwargs)  # type: ignore[arg-type]

    def subdivide_tetra(self):
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

        Now, subdivide and plot.

        >>> subdivided = grid.subdivide_tetra()
        >>> subdivided.plot(show_edges=True, line_width=2)

        """
        alg = _vtk.vtkSubdivideTetra()
        alg.SetInputData(self)
        _update_alg(alg)
        return _get_output(alg)

    @_deprecate_positional_args
    def clean(  # noqa: PLR0917
        self,
        tolerance=0,
        remove_unused_points: bool = True,  # noqa: FBT001, FBT002
        produce_merge_map: bool = True,  # noqa: FBT001, FBT002
        average_point_data: bool = True,  # noqa: FBT001, FBT002
        merging_array_name=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ):
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
        try:
            from vtkmodules.vtkFiltersCore import vtkStaticCleanUnstructuredGrid  # noqa: PLC0415
        except ImportError:  # pragma no cover
            msg = 'UnstructuredGrid.clean requires VTK >= 9.2.2'
            raise VTKVersionError(msg) from None

        alg = vtkStaticCleanUnstructuredGrid()
        # https://github.com/pyvista/pyvista/pull/6337
        alg.SetInputDataObject(self.copy())  # type: ignore[attr-defined]
        alg.SetAbsoluteTolerance(True)
        alg.SetTolerance(tolerance)
        alg.SetMergingArray(merging_array_name)
        alg.SetRemoveUnusedPoints(remove_unused_points)
        alg.SetProduceMergeMap(produce_merge_map)
        alg.SetAveragePointData(average_point_data)
        _update_alg(alg, progress_bar=progress_bar, message='Cleaning Unstructured Grid')
        return _get_output(alg)

    def remove_unused_points(  # type: ignore[misc]
        self: UnstructuredGrid,
        *,
        inplace: bool = False,
    ):
        """Remove points which are not used by any cells.

        Unlike :meth:`clean`, this filter does `not` merge points. The point order is also
        unchanged by this filter.

        .. note::
            This filter is inefficient. If point merging is acceptable, :meth:`clean` should
            be used instead.

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

        def _add_vertex_cell_to_unused_points(mesh):
            # Find unused point indices
            merge_map = mesh.clean(
                remove_unused_points=True,
                produce_merge_map=True,
                average_point_data=False,
            ).field_data['PointMergeMap']
            unused_point_ids = np.where(merge_map == -1)[0]

            # Create vertex cells [1, pt_id] for each unused point
            n_unused = len(unused_point_ids)
            vertex_cells = np.empty(n_unused * 2, dtype=np.int32)
            vertex_cells[0::2] = 1
            vertex_cells[1::2] = unused_point_ids

            # Concatenate with original cell array and cell types
            all_cells = np.concatenate([mesh.cells, vertex_cells])
            all_celltypes = np.concatenate(
                [
                    mesh.celltypes,
                    np.full(n_unused, CellType.VERTEX, dtype=np.uint8),
                ]
            )

            # Convert to vtk arrays and set the new cells
            vtk_cells = CellArray(all_cells)
            vtk_celltypes = _vtk.numpy_to_vtk(all_celltypes)
            mesh.SetCells(vtk_celltypes, vtk_cells)

        # Use extract_points to only keep point IDs associated with cells.
        # Surprisingly, extract_points will keep unused points, even if their point IDs are
        # not included, so we first need to explicitly map unused points to VERTEX cells
        new_grid = self.copy(deep=False)
        if not new_grid.is_empty:
            _add_vertex_cell_to_unused_points(new_grid)
            new_grid = new_grid.extract_points(self.cell_connectivity)

            if (name := 'vtkOriginalPointIds') in (data := new_grid.point_data):
                del data[name]
            if (name := 'vtkOriginalCellIds') in (data := new_grid.cell_data):
                del data[name]

        if inplace:
            self.copy_from(new_grid, deep=False)
            return self
        return new_grid
