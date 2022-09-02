import numpy as np
import pytest
import vtk
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


def test_empty():
    grid = pyvista.cells.Empty()
    assert grid.celltypes[0] == pyvista.CellType.EMPTY_CELL
    assert grid.n_cells == 1
    assert grid.n_points == 0


def test_vertex():
    grid = pyvista.cells.Vertex()
    assert grid.celltypes[0] == pyvista.CellType.VERTEX
    assert grid.n_cells == 1
    assert grid.n_points == 1


def test_poly_vertex():
    grid = pyvista.cells.PolyVertex()
    assert grid.celltypes[0] == pyvista.CellType.POLY_VERTEX
    assert grid.n_cells == 1
    assert grid.n_points >= 1


def test_line():
    grid = pyvista.cells.Line()
    assert grid.celltypes[0] == pyvista.CellType.LINE
    assert grid.n_cells == 1
    assert grid.n_points == 2


def test_poly_line():
    grid = pyvista.cells.PolyLine()
    assert grid.celltypes[0] == pyvista.CellType.POLY_LINE
    assert grid.n_cells == 1
    assert grid.n_points >= 1


def test_triangle():
    grid = pyvista.cells.Triangle()
    assert grid.celltypes[0] == pyvista.CellType.TRIANGLE
    assert grid.n_cells == 1
    assert grid.n_points == 3


def test_triangle_strip():
    grid = pyvista.cells.TriangleStrip()
    assert grid.celltypes[0] == pyvista.CellType.TRIANGLE_STRIP
    assert grid.n_cells == 1
    assert grid.n_points >= 3


def test_polygon():
    grid = pyvista.cells.Polygon()
    assert grid.celltypes[0] == pyvista.CellType.POLYGON
    assert grid.n_cells == 1
    assert grid.n_points == 6


def test_Quadrilateral():
    grid = pyvista.cells.Quadrilateral()
    assert grid.celltypes[0] == pyvista.CellType.QUAD
    assert grid.n_cells == 1
    assert grid.n_points == 4


def test_tetrahedron():
    grid = pyvista.cells.Tetrahedron()
    assert grid.celltypes[0] == pyvista.CellType.TETRA
    assert grid.n_cells == 1
    assert grid.n_points == 4


def test_hexagonal_prism():
    grid = pyvista.cells.HexagonalPrism()
    assert grid.celltypes[0] == pyvista.CellType.HEXAGONAL_PRISM
    assert grid.n_cells == 1
    assert grid.n_points == 12


def test_hexahedron():
    grid = pyvista.cells.Hexahedron()
    assert grid.celltypes[0] == pyvista.CellType.HEXAHEDRON
    assert grid.n_cells == 1
    assert grid.n_points == 8


def test_pyramid():
    grid = pyvista.cells.Pyramid()
    assert grid.celltypes[0] == pyvista.CellType.PYRAMID
    assert grid.n_cells == 1
    assert grid.n_points == 5


def test_wedge():
    grid = pyvista.cells.Wedge()
    assert grid.celltypes[0] == pyvista.CellType.WEDGE
    assert grid.n_cells == 1
    assert grid.n_points == 6


def test_pentagonal_prism():
    grid = pyvista.cells.PentagonalPrism()
    assert grid.celltypes[0] == pyvista.CellType.PENTAGONAL_PRISM
    assert grid.n_cells == 1
    assert grid.n_points == 10


def test_voxel():
    grid = pyvista.cells.Voxel()
    assert grid.celltypes[0] == pyvista.CellType.VOXEL
    assert grid.n_cells == 1
    assert grid.n_points == 8


def test_pixel():
    grid = pyvista.cells.Pixel()
    assert grid.celltypes[0] == pyvista.CellType.PIXEL
    assert grid.n_cells == 1
    assert grid.n_points == 4
