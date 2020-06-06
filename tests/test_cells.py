import pytest
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

import pyvista

CELL_LIST = [3, 0, 1, 2, 3, 3, 4, 5]
NCELLS = 2
FCONTIG_ARR = np.array(np.vstack(([3, 0, 1, 2], [3, 3, 4, 5])), order='F')


@pytest.mark.parametrize('deep', [False, True])
@pytest.mark.parametrize('n_cells', [None, NCELLS])
@pytest.mark.parametrize('cells', [CELL_LIST,
                                   np.array(CELL_LIST, np.int16),
                                   np.array(CELL_LIST, np.int32),
                                   np.array(CELL_LIST, np.int64),
                                   FCONTIG_ARR])
def test_init_cell_array(cells, n_cells, deep):
    cell_array = pyvista.utilities.cells.CellArray(cells, n_cells, deep)
    assert np.allclose(np.array(cells).ravel(), cell_array.cells)
    assert cell_array.n_cells == cell_array.GetNumberOfCells() == NCELLS


def test_numpy_to_idarr_bool():
    mask = np.ones(10, np.bool_)
    idarr = pyvista.utilities.cells.numpy_to_idarr(mask)
    assert np.allclose(mask.nonzero()[0], vtk_to_numpy(idarr))
