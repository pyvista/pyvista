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
