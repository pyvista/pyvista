"""Filters module with a class to manage filters/algorithms for unstructured grid datasets."""
from functools import wraps

from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import VTKVersionError
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.filters.poly_data import PolyDataFilters
from pyvista.core.utilities.misc import abstract_class


@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

    @wraps(PolyDataFilters.delaunay_2d)
    def delaunay_2d(self, *args, **kwargs):  # numpydoc ignore=PR01,RT01
        """Wrap ``PolyDataFilters.delaunay_2d``."""
        return PolyDataFilters.delaunay_2d(self, *args, **kwargs)

    @wraps(PolyDataFilters.reconstruct_surface)
    def reconstruct_surface(self, *args, **kwargs):  # numpydoc ignore=PR01,RT01
        """Wrap ``PolyDataFilters.reconstruct_surface``."""
        return PolyDataFilters.reconstruct_surface(self, *args, **kwargs)

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

    def clean(
        self,
        tolerance=0,
        remove_unused_points=True,
        produce_merge_map=True,
        average_point_data=True,
        merging_array_name=None,
        progress_bar=False,
    ):
        """Merge duplicate points and remove unused points in an UnstructuredGrid.

        This filter, merging coincident points as defined by a merging
        tolerance and optionally removes unused points. The filter does not
        modify the topology of the input dataset, nor change the types of
        cells. It may however, renumber the cell connectivity ids.

        This filter implements `vtkStaticCleanUnstructuredGrid
        <https://vtk.org/doc/nightly/html/classvtkStaticCleanUnstructuredGrid.html>`_

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
            from vtkmodules.vtkFiltersCore import vtkStaticCleanUnstructuredGrid
        except ImportError:  # pragma no cover
            raise VTKVersionError("UnstructuredGrid.clean requires VTK >= 9.2.2") from None

        alg = vtkStaticCleanUnstructuredGrid()
        alg.SetInputDataObject(self)
        alg.SetAbsoluteTolerance(True)
        alg.SetTolerance(tolerance)
        alg.SetMergingArray(merging_array_name)
        alg.SetRemoveUnusedPoints(remove_unused_points)
        alg.SetProduceMergeMap(produce_merge_map)
        alg.SetAveragePointData(average_point_data)
        _update_alg(alg, progress_bar, 'Cleaning Unstructured Grid')
        return _get_output(alg)
