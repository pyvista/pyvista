"""Filters module with a class to manage filters/algorithms for unstructured grid datasets."""
from functools import wraps

import numpy as np

from pyvista import _vtk, abstract_class
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.filters.poly_data import PolyDataFilters


@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

    def clean(self, progress_bar=False):
        """Remove redundant or unused cells or points.

        This also removes all zero data arrays.

        Parameters
        ----------
        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.UnstructuredGrid
            Cleaned unstructured grid with additional ``'OriginalPointIds```
            array added to the dataset indicating the relationship between the
            cleaned points and the original points.

        Notes
        -----
        Requires ``vtk>=9.0.3``.

        Examples
        --------
        Merge two adjacent grids without merging their points.

        >>> from pyvista import examples
        >>> grid_a = examples.load_hexbeam()
        >>> grid_b = grid_a.copy()
        >>> grid_b.points[:, 0] += 1
        >>> grid_c = grid_a.merge(grid_b, merge_points=False)
        >>> grid_c  # doctest:+SKIP
        >>> grid_c.n_points
        198

        Now, clean the grid and show the overlapping points have been merged.

        >>> grid_merged = grid_c.clean()
        >>> grid_merged.n_points
        165

        """
        try:  # pragma: no cover
            from vtkmodules.vtkAcceleratorsVTKm import vtkmCleanGrid
        except ImportError:
            try:
                from vtkmodules.vtkAcceleratorsVTKmFilters import vtkmCleanGrid
            except ImportError:
                raise ImportError('Install vtk>=9.0.3 for this feature') from None

        point_id_name = 'OriginalPointIds'
        self.point_data[point_id_name] = np.arange(self.n_points, dtype=np.int32)

        alg = vtkmCleanGrid()
        alg.SetInputDataObject(self)
        # Must set compact points to true. Otherwise, we risk a segfault since
        # the number of points in each point_data array may not match the
        # number of points in the output grid.
        alg.SetCompactPoints(True)
        _update_alg(alg, progress_bar=progress_bar, message='Cleaning UnstructuredGrid')
        output = _get_output(alg)

        # Not all arrays are preserved. In particular, arrays with more than 3
        # components seem to be removed
        # ids = output.point_data[point_id_name]
        # for key in self.point_data:
        #     if key not in output.point_data:
        #         output.point_data[key] = self.point_data[key][ids]

        self.point_data[point_id_name]
        return output

    @wraps(PolyDataFilters.delaunay_2d)
    def delaunay_2d(self, *args, **kwargs):
        """Wrap ``PolyDataFilters.delaunay_2d``."""
        return PolyDataFilters.delaunay_2d(self, *args, **kwargs)

    @wraps(PolyDataFilters.reconstruct_surface)
    def reconstruct_surface(self, *args, **kwargs):
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
