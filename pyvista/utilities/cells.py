"""pyvista wrapping of vtkCellArray."""
import numpy as np
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtk import vtkCellArray

import pyvista


def numpy_to_idarr(ind, deep=False, return_ind=False):
    """Safely convert a numpy array to a vtkIdTypeArray."""
    ind = np.asarray(ind)

    # np.asarray will eat anything, so we have to weed out bogus inputs
    if not issubclass(ind.dtype.type, (np.bool_, np.integer)):
        raise TypeError('Indices must be either a mask or an integer array-like')

    if ind.dtype == np.bool_:
        ind = ind.nonzero()[0].astype(pyvista.ID_TYPE)
    elif ind.dtype != pyvista.ID_TYPE:
        ind = ind.astype(pyvista.ID_TYPE)
    elif not ind.flags['C_CONTIGUOUS']:
        ind = np.ascontiguousarray(ind, dtype=pyvista.ID_TYPE)

    # must ravel or segfault when saving MultiBlock
    vtk_idarr = numpy_to_vtkIdTypeArray(ind.ravel(), deep=deep)
    if return_ind:
        return vtk_idarr, ind
    return vtk_idarr


class CellArray(vtkCellArray):
    """pyvista wrapping of vtkCellArray.

    Provides convenience functions to simplify creating a CellArray from
    a numpy array or list.

    Import an array of data with the legacy vtkCellArray layout, e.g.

    ``{ n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ... }``
    Where n0 is the number of points in cell 0, and pX_Y is the Y'th
    point in cell X.

    Examples
    --------
    Create a cell array containing two triangles.
    >>> from pyvista.utilities.cells import CellArray
    >>> cellarr = CellArray([3, 0, 1, 2, 3, 3, 4, 5])
    """

    def __init__(self, cells=None, n_cells=None, deep=False):
        """Initialize a vtkCellArray."""
        if cells is not None:
            self._set_cells(cells, n_cells, deep)

    def _set_cells(self, cells, n_cells, deep):
        vtk_idarr, cells = numpy_to_idarr(cells, deep=deep, return_ind=True)
        # get number of cells if none
        if n_cells is None:
            if cells.ndim == 1:
                c = 0
                n_cells = 0
                while c < cells.size:
                    c += cells[c] + 1
                    n_cells += 1
            else:
                n_cells = cells.shape[0]

        self.SetCells(n_cells, vtk_idarr)

    @property
    def cells(self):
        """Return a numpy array of the cells."""
        return vtk_to_numpy(self.GetData()).ravel()

    @property
    def n_cells(self):
        """Return the number of cells."""
        return self.GetNumberOfCells()
