"""Filters module with a class to manage filters/algorithms for unstructured grid datasets."""
from functools import wraps

import pyvista
from pyvista import abstract_class
from pyvista.core.filters.poly_data import PolyDataFilters
from pyvista.core.filters.data_set import DataSetFilters


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
