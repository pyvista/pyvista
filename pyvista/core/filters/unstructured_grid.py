"""Filters module with a class to manage filters/algorithms for unstructured grid datasets."""
import pyvista
from pyvista import abstract_class
from pyvista.core.filters.data_set import DataSetFilters


@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

    def delaunay_2d(self, tol=1e-05, alpha=0.0, offset=1.0, bound=False,
                    progress_bar=False):
        """Apply a delaunay 2D filter along the best fitting plane.

        This extracts the grid's points and performs the triangulation
        on those alone.

        Parameters
        ----------
        tol : float, optional
            Specify a tolerance to control discarding of closely
            spaced points. This tolerance is specified as a fraction
            of the diagonal length of the bounding box of the points.
            Defaults to ``1e-05``.

        alpha : float, optional
            Specify alpha (or distance) value to control output of
            this filter. For a non-zero alpha value, only edges or
            triangles contained within a sphere centered at mesh
            vertices will be output. Otherwise, only triangles will be
            output. Defaults to ``0.0``.

        offset : float, optional
            Specify a multiplier to control the size of the initial,
            bounding Delaunay triangulation. Defaults to ``1.0``.

        bound : bool, optional
            Boolean controls whether bounding triangulation points
            and associated triangles are included in the
            output. These are introduced as an initial triangulation
            to begin the triangulation process. This feature is nice
            for debugging output. Default ``False``.

        progress_bar : bool, optional
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Surface mesh containing the delaunay filter.

        """
        return pyvista.PolyData(self.points).delaunay_2d(tol=tol, alpha=alpha,
                                                         offset=offset,
                                                         bound=bound,
                                                         progress_bar=progress_bar)
