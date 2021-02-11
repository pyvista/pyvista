"""These classes hold methods to apply general filters to any data type.

By inheriting these classes into the wrapped VTK data structures, a user
can easily apply common filters in an intuitive manner.

Example
-------
>>> import pyvista
>>> from pyvista import examples
>>> dataset = examples.load_uniform()

>>> # Threshold
>>> thresh = dataset.threshold([100, 500])

>>> # Slice
>>> slc = dataset.slice()

>>> # Clip
>>> clp = dataset.clip(invert=True)

>>> # Contour
>>> iso = dataset.contour()

"""

from . import DataSetFilters
import pyvista
from pyvista.utilities import (abstract_class)


@abstract_class
class UnstructuredGridFilters(DataSetFilters):
    """An internal class to manage filters/algorithms for unstructured grid datasets."""

    def delaunay_2d(ugrid, tol=1e-05, alpha=0.0, offset=1.0, bound=False,
                    progress_bar=False):
        """Apply a delaunay 2D filter along the best fitting plane.

        This extracts the grid's points and performs the triangulation on those alone.

        Parameters
        ----------
        progress_bar : bool, optional
            Display a progress bar to indicate progress.
        """
        return pyvista.PolyData(ugrid.points).delaunay_2d(tol=tol, alpha=alpha,
                                                          offset=offset,
                                                          bound=bound,
                                                          progress_bar=progress_bar)
