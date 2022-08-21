import numpy as np
import pytest
from vtk.util.numpy_support import vtk_to_numpy

import pyvista

CELL_LIST = [3, 0, 1, 2, 3, 3, 4, 5]
NCELLS = 2
FCONTIG_ARR = np.array(np.vstack(([3, 0, 1, 2], [3, 3, 4, 5])), order='F')


@pytest.mark.parametrize('deep', [False, True])
@pytest.mark.parametrize('n_cells', [None, NCELLS])
@pytest.mark.parametrize(
    'cells',
    [
        CELL_LIST,
        np.array(CELL_LIST, np.int16),
        np.array(CELL_LIST, np.int32),
        np.array(CELL_LIST, np.int64),
        FCONTIG_ARR,
    ],
)
def test_init_cell_array(cells, n_cells, deep):
    cell_array = pyvista.utilities.cells.CellArray(cells, n_cells, deep)
    assert np.allclose(np.array(cells).ravel(), cell_array.cells)
    assert cell_array.n_cells == cell_array.GetNumberOfCells() == NCELLS


def test_numpy_to_idarr_bool():
    mask = np.ones(10, np.bool_)
    idarr = pyvista.utilities.cells.numpy_to_idarr(mask)
    assert np.allclose(mask.nonzero()[0], vtk_to_numpy(idarr))


def test_cell_types():
    cell_types = [
        "EMPTY_CELL",
        "VERTEX",
        "POLY_VERTEX",
        "LINE",
        "POLY_LINE",
        "TRIANGLE",
        "TRIANGLE_STRIP",
        "POLYGON",
        "PIXEL",
        "QUAD",
        "TETRA",
        "VOXEL",
        "HEXAHEDRON",
        "WEDGE",
        "PYRAMID",
        "PENTAGONAL_PRISM",
        "HEXAGONAL_PRISM",
        "QUADRATIC_EDGE",
        "QUADRATIC_TRIANGLE",
        "QUADRATIC_QUAD",
        "QUADRATIC_POLYGON",
        "QUADRATIC_TETRA",
        "QUADRATIC_HEXAHEDRON",
        "QUADRATIC_WEDGE",
        "QUADRATIC_PYRAMID",
        "BIQUADRATIC_QUAD",
        "TRIQUADRATIC_HEXAHEDRON",
        "TRIQUADRATIC_PYRAMID",
        "QUADRATIC_LINEAR_QUAD",
        "QUADRATIC_LINEAR_WEDGE",
        "BIQUADRATIC_QUADRATIC_WEDGE",
        "BIQUADRATIC_QUADRATIC_HEXAHEDRON",
        "BIQUADRATIC_TRIANGLE",
        "CUBIC_LINE",
        "CONVEX_POINT_SET",
        "POLYHEDRON",
        "PARAMETRIC_CURVE",
        "PARAMETRIC_SURFACE",
        "PARAMETRIC_TRI_SURFACE",
        "PARAMETRIC_QUAD_SURFACE",
        "PARAMETRIC_TETRA_REGION",
        "PARAMETRIC_HEX_REGION",
        "HIGHER_ORDER_EDGE",
        "HIGHER_ORDER_TRIANGLE",
        "HIGHER_ORDER_QUAD",
        "HIGHER_ORDER_POLYGON",
        "HIGHER_ORDER_TETRAHEDRON",
        "HIGHER_ORDER_WEDGE",
        "HIGHER_ORDER_PYRAMID",
        "HIGHER_ORDER_HEXAHEDRON",
        "LAGRANGE_CURVE",
        "LAGRANGE_TRIANGLE",
        "LAGRANGE_QUADRILATERAL",
        "LAGRANGE_TETRAHEDRON",
        "LAGRANGE_HEXAHEDRON",
        "LAGRANGE_WEDGE",
        "LAGRANGE_PYRAMID",
        "BEZIER_CURVE",
        "BEZIER_TRIANGLE",
        "BEZIER_QUADRILATERAL",
        "BEZIER_TETRAHEDRON",
        "BEZIER_HEXAHEDRON",
        "BEZIER_WEDGE",
        "BEZIER_PYRAMID",
    ]
    for cell_type in cell_types:
        if hasattr(vtk, "VTK_" + cell_type):
            assert getattr(pyvista.CellType, cell_type) == getattr(vtk, 'VTK_' + cell_type)
