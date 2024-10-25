from __future__ import annotations

import pytest

from pyvista import CellType
from pyvista.examples import cells


def test_empty():
    grid = cells.Empty()
    assert grid.celltypes[0] == CellType.EMPTY_CELL
    assert grid.n_cells == 1
    assert grid.n_points == 0


def test_vertex():
    grid = cells.Vertex()
    assert grid.celltypes[0] == CellType.VERTEX
    assert grid.n_cells == 1
    assert grid.n_points == 1


def test_poly_vertex():
    grid = cells.PolyVertex()
    assert grid.celltypes[0] == CellType.POLY_VERTEX
    assert grid.n_cells == 1
    assert grid.n_points >= 1


def test_line():
    grid = cells.Line()
    assert grid.celltypes[0] == CellType.LINE
    assert grid.n_cells == 1
    assert grid.n_points == 2


def test_poly_line():
    grid = cells.PolyLine()
    assert grid.celltypes[0] == CellType.POLY_LINE
    assert grid.n_cells == 1
    assert grid.n_points >= 1


def test_triangle():
    grid = cells.Triangle()
    assert grid.celltypes[0] == CellType.TRIANGLE
    assert grid.n_cells == 1
    assert grid.n_points == 3


def test_triangle_strip():
    grid = cells.TriangleStrip()
    assert grid.celltypes[0] == CellType.TRIANGLE_STRIP
    assert grid.n_cells == 1
    assert grid.n_points >= 3


def test_polygon():
    grid = cells.Polygon()
    assert grid.celltypes[0] == CellType.POLYGON
    assert grid.n_cells == 1
    assert grid.n_points == 6


def test_Quadrilateral():
    grid = cells.Quadrilateral()
    assert grid.celltypes[0] == CellType.QUAD
    assert grid.n_cells == 1
    assert grid.n_points == 4


def test_tetrahedron():
    grid = cells.Tetrahedron()
    assert grid.celltypes[0] == CellType.TETRA
    assert grid.n_cells == 1
    assert grid.n_points == 4


def test_hexagonal_prism():
    grid = cells.HexagonalPrism()
    assert grid.celltypes[0] == CellType.HEXAGONAL_PRISM
    assert grid.n_cells == 1
    assert grid.n_points == 12


def test_hexahedron():
    grid = cells.Hexahedron()
    assert grid.celltypes[0] == CellType.HEXAHEDRON
    assert grid.n_cells == 1
    assert grid.n_points == 8


def test_pyramid():
    grid = cells.Pyramid()
    assert grid.celltypes[0] == CellType.PYRAMID
    assert grid.n_cells == 1
    assert grid.n_points == 5


def test_wedge():
    grid = cells.Wedge()
    assert grid.celltypes[0] == CellType.WEDGE
    assert grid.n_cells == 1
    assert grid.n_points == 6


def test_pentagonal_prism():
    grid = cells.PentagonalPrism()
    assert grid.celltypes[0] == CellType.PENTAGONAL_PRISM
    assert grid.n_cells == 1
    assert grid.n_points == 10


def test_voxel():
    grid = cells.Voxel()
    assert grid.celltypes[0] == CellType.VOXEL
    assert grid.n_cells == 1
    assert grid.n_points == 8


def test_pixel():
    grid = cells.Pixel()
    assert grid.celltypes[0] == CellType.PIXEL
    assert grid.n_cells == 1
    assert grid.n_points == 4


def test_quadratic_edge():
    grid = cells.QuadraticEdge()
    assert grid.celltypes[0] == CellType.QUADRATIC_EDGE
    assert grid.n_cells == 1
    assert grid.n_points == 3


def test_quadratic_triangle():
    grid = cells.QuadraticTriangle()
    assert grid.celltypes[0] == CellType.QUADRATIC_TRIANGLE
    assert grid.n_cells == 1
    assert grid.n_points == 6


def test_quadratic_quadrilateral():
    grid = cells.QuadraticQuadrilateral()
    assert grid.celltypes[0] == CellType.QUADRATIC_QUAD
    assert grid.n_cells == 1
    assert grid.n_points == 8


def test_quadratic_tetrahedron():
    grid = cells.QuadraticTetrahedron()
    assert grid.celltypes[0] == CellType.QUADRATIC_TETRA
    assert grid.n_cells == 1
    assert grid.n_points == 10


def test_quadratic_hexahedron():
    grid = cells.QuadraticHexahedron()
    assert grid.celltypes[0] == CellType.QUADRATIC_HEXAHEDRON
    assert grid.n_cells == 1
    assert grid.n_points == 20


def test_quadratic_wedge():
    grid = cells.QuadraticWedge()
    assert grid.celltypes[0] == CellType.QUADRATIC_WEDGE
    assert grid.n_cells == 1
    assert grid.n_points == 15


def test_quadratic_polygon():
    grid = cells.QuadraticPolygon()
    assert grid.celltypes[0] == CellType.QUADRATIC_POLYGON
    assert grid.n_cells == 1
    assert grid.n_points == 8


def test_quadratic_pyramid():
    grid = cells.QuadraticPyramid()
    assert grid.celltypes[0] == CellType.QUADRATIC_PYRAMID
    assert grid.n_cells == 1
    assert grid.n_points == 13


def test_biquadratic_quadrilateral():
    grid = cells.BiQuadraticQuadrilateral()
    assert grid.celltypes[0] == CellType.BIQUADRATIC_QUAD
    assert grid.n_cells == 1
    assert grid.n_points == 9


def test_triquadratic_hexahedron():
    grid = cells.TriQuadraticHexahedron()
    assert grid.celltypes[0] == CellType.TRIQUADRATIC_HEXAHEDRON
    assert grid.n_cells == 1
    assert grid.n_points == 27


@pytest.mark.needs_vtk_version(9, 1, 0)
def test_triquadratic_pyramid():
    grid = cells.TriQuadraticPyramid()
    assert grid.celltypes[0] == CellType.TRIQUADRATIC_PYRAMID
    assert grid.n_cells == 1
    assert grid.n_points == 19


def test_quadratic_linear_quadrilateral():
    grid = cells.QuadraticLinearQuadrilateral()
    assert grid.celltypes[0] == CellType.QUADRATIC_LINEAR_QUAD
    assert grid.n_cells == 1
    assert grid.n_points == 6


def test_quadratic_linear_wedge():
    grid = cells.QuadraticLinearWedge()
    assert grid.celltypes[0] == CellType.QUADRATIC_LINEAR_WEDGE
    assert grid.n_cells == 1
    assert grid.n_points == 12


def test_biquadratic_quadratic_wedge():
    grid = cells.BiQuadraticQuadraticWedge()
    assert grid.celltypes[0] == CellType.BIQUADRATIC_QUADRATIC_WEDGE
    assert grid.n_cells == 1
    assert grid.n_points == 18


def test_biquadratic_quadratic_hexahedron():
    grid = cells.BiQuadraticQuadraticHexahedron()
    assert grid.celltypes[0] == CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON
    assert grid.n_cells == 1
    assert grid.n_points == 24


def test_biquadratic_triangle():
    grid = cells.BiQuadraticTriangle()
    assert grid.celltypes[0] == CellType.BIQUADRATIC_TRIANGLE
    assert grid.n_cells == 1
    assert grid.n_points == 7


def test_cubic_line():
    grid = cells.CubicLine()
    assert grid.celltypes[0] == CellType.CUBIC_LINE
    assert grid.n_cells == 1
    assert grid.n_points == 4
