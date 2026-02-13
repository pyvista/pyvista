from __future__ import annotations

import inspect

import numpy as np
import pytest
from pytest_cases import parametrize

import pyvista as pv
from pyvista import CellType
from pyvista.examples import cells

# Collect all functions in the cells module that start with a capital letter
cell_example_functions = [
    func for name, func in inspect.getmembers(cells, inspect.isfunction) if name[0].isupper()
]


@parametrize('cell_example', cell_example_functions)
def test_area_and_volume(cell_example):
    mesh = cell_example()
    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.n_cells == 1

    # Volume should be positive but returns zero or negative, see https://gitlab.kitware.com/vtk/vtk/-/issues/19639
    ctype = mesh.celltypes[0]
    if ctype == CellType.QUADRATIC_WEDGE:
        assert mesh.volume < 0
        return
    elif ctype == CellType.TRIQUADRATIC_HEXAHEDRON:
        if pv.vtk_version_info >= (9, 6, 0):
            assert mesh.volume < 0
        elif pv.vtk_version_info < (9, 4, 0):
            assert mesh.volume == 0
        return
    elif ctype == CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON and pv.vtk_version_info < (9, 4, 0):
        assert mesh.volume == 0
        return

    # Test area and volume
    dim = mesh.GetCell(0).GetCellDimension()
    area = mesh.area
    volume = mesh.volume
    if dim == 2:
        assert area > 0
        assert np.isclose(volume, 0.0)
    elif dim == 3:
        assert np.isclose(area, 0.0)
        assert volume > 0
    else:
        assert np.isclose(area, 0.0)
        assert np.isclose(volume, 0.0)


@pytest.mark.needs_vtk_version(9, 5, 0, reason='vtkCellValidator output differs')
@parametrize('cell_example', cell_example_functions)
def test_cell_is_valid(cell_example):
    mesh = cell_example()
    invalid_fields = mesh.validate_mesh().invalid_fields
    cell_type = next(mesh.cell).type
    if cell_type == pv.CellType.QUADRATIC_WEDGE or (
        cell_type == pv.CellType.TRIQUADRATIC_HEXAHEDRON and pv.vtk_version_info >= (9, 6, 0)
    ):
        # Caused by negative volume bug https://gitlab.kitware.com/vtk/vtk/-/issues/19639
        assert invalid_fields == ('negative_size',)
    else:
        assert not invalid_fields


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
    assert np.isclose(grid.area, np.sqrt(3) / 4)


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


def test_quadrilateral():
    grid = cells.Quadrilateral()
    assert grid.celltypes[0] == CellType.QUAD
    assert grid.n_cells == 1
    assert grid.n_points == 4
    assert np.isclose(grid.area, 1.0)


def test_tetrahedron():
    grid = cells.Tetrahedron()
    assert grid.celltypes[0] == CellType.TETRA
    assert grid.n_cells == 1
    assert grid.n_points == 4
    assert np.isclose(grid.volume, np.sqrt(2) / 12)


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
    assert np.isclose(grid.volume, 1.0)


def test_pyramid():
    grid = cells.Pyramid()
    assert grid.celltypes[0] == CellType.PYRAMID
    assert grid.n_cells == 1
    assert grid.n_points == 5
    assert np.isclose(grid.volume, np.sqrt(2.0) / 6.0)


def test_wedge():
    grid = cells.Wedge()
    assert grid.celltypes[0] == CellType.WEDGE
    assert grid.n_cells == 1
    assert grid.n_points == 6
    assert np.isclose(grid.volume, np.sqrt(3.0) / 4.0)


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
