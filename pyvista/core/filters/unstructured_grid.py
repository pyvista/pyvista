"""Filters module with a class to manage filters/algorithms for unstructured grid datasets."""
from functools import wraps

from pyvista import _vtk, abstract_class
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.filters.poly_data import PolyDataFilters


@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

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
