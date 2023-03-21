from types import GeneratorType

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from pyvista import Cell, CellType
from pyvista.examples import (
    cells as example_cells,
    load_airplane,
    load_explicit_structured,
    load_hexbeam,
    load_rectilinear,
    load_structured,
    load_tetbeam,
    load_uniform,
)

cells = [
    example_cells.Hexahedron().get_cell(0),
    example_cells.Triangle().get_cell(0),
    example_cells.Voxel().get_cell(0),
    example_cells.Quadrilateral().get_cell(0),
    example_cells.Tetrahedron().get_cell(0),
    example_cells.Voxel().get_cell(0),
]
grids = [
    load_hexbeam(),
    load_airplane(),
    load_rectilinear(),
    load_structured(),
    load_tetbeam(),
    load_uniform(),
    load_explicit_structured(),
]
types = [
    CellType.HEXAHEDRON,
    CellType.TRIANGLE,
    CellType.VOXEL,
    CellType.QUAD,
    CellType.TETRA,
    CellType.VOXEL,
    CellType.HEXAHEDRON,
]
faces_types = [
    CellType.QUAD,
    None,
    CellType.PIXEL,
    None,
    CellType.TRIANGLE,
    CellType.PIXEL,
    CellType.QUAD,
]
dims = [3, 2, 3, 2, 3, 3, 3]
npoints = [8, 3, 8, 4, 4, 8, 8]
nfaces = [6, 0, 6, 0, 4, 6, 6]
nedges = [12, 3, 12, 4, 6, 12, 12, 12]

ids = [str(type(grid)) for grid in grids]
cell_ids = list(map(repr, types))


def test_bad_init():
    with pytest.raises(TypeError, match="must be a vtkCell"):
        _ = Cell(1)


@pytest.mark.parametrize("grid", grids, ids=ids)
def test_cell_attribute(grid):
    assert isinstance(grid.cell, GeneratorType)
    assert all([issubclass(type(cell), Cell) for cell in grid.cell])


@pytest.mark.parametrize("grid", grids, ids=ids)
def test_cell_point_ids(grid):
    # Test that the point_ids for all cells in the grid are unique,
    # which is not the case when using the GetCell(i) method of DataSet.
    # See https://vtk.org/doc/nightly/html/classvtkDataSet.html#a711ed1ebb7bdf4a4e2ed6896081cd1b2
    point_ids = {frozenset(c.point_ids) for c in grid.cell}
    assert len(point_ids) == grid.n_cells


def test_cell_get_cell():
    hexbeam = grids[0]
    with pytest.raises(IndexError, match='Invalid index'):
        hexbeam.get_cell(hexbeam.n_cells)
    assert isinstance(hexbeam.get_cell(0), pyvista.Cell)


@pytest.mark.parametrize("cell", cells[:5])
def test_cell_type_is_inside_enum(cell):
    assert cell.type in CellType


@pytest.mark.parametrize("cell,type", zip(cells[:5], types[:5]))
def test_cell_type(cell, type):
    assert cell.type == type


@pytest.mark.parametrize("cell", cells)
def test_cell_is_linear(cell):
    assert cell.is_linear


@pytest.mark.parametrize("cell, dim", zip(cells[:5], dims[:5]))
def test_cell_dimension(cell, dim):
    assert cell.dimension == dim


@pytest.mark.parametrize("cell, np", zip(cells[:5], npoints[:5]))
def test_cell_n_points(cell, np):
    assert cell.n_points == np


@pytest.mark.parametrize("cell, nf", zip(cells[:5], nfaces[:5]))
def test_cell_n_faces(cell, nf):
    assert cell.n_faces == nf


@pytest.mark.parametrize("cell, ne", zip(cells[:5], nedges[:5]))
def test_cell_n_edges(cell, ne):
    assert cell.n_edges == ne


@pytest.mark.parametrize("cell", cells[:5])
def test_cell_get_edges(cell):
    assert all(cell.get_edge(i).type == CellType.LINE for i in range(cell.n_edges))

    with pytest.raises(IndexError, match='Invalid index'):
        cell.get_edge(cell.n_edges)


@pytest.mark.parametrize("cell", cells[:5])
def test_cell_edges(cell):
    assert all(edge.type == CellType.LINE for edge in cell.edges)


def test_cell_no_field_data():
    with pytest.raises(NotImplementedError, match='does not support field data'):
        cells[0].add_field_data([1, 2, 3], 'field_data')

    with pytest.raises(NotImplementedError, match='does not support field data'):
        cells[0].clear_field_data()


@pytest.mark.parametrize("cell", cells[:5])
def test_cell_copy_generic(cell):
    cell = cell.copy()
    cell_copy = cell.copy(deep=True)
    assert cell_copy == cell
    cell_copy.points[:] = 0
    assert cell_copy != cell

    cell_copy = cell.copy(deep=False)
    assert cell_copy == cell
    cell_copy.points[:] = 0
    assert cell_copy == cell


def test_cell_copy():
    cell = cells[0].get_face(0)
    assert isinstance(cell, pyvista.Cell)
    cell_copy = cell.copy(deep=True)
    assert cell_copy == cell
    cell_copy.points[:] = 0
    assert cell_copy != cell

    cell_copy = cell.copy(deep=False)
    assert cell_copy == cell
    cell_copy.points[:] = 0
    assert cell_copy == cell


@pytest.mark.parametrize("cell", cells[:5])
def test_cell_edges_point_ids(cell):
    point_ids = {frozenset(cell.get_edge(i).point_ids) for i in range(cell.n_edges)}
    assert len(point_ids) == cell.n_edges


@pytest.mark.parametrize("cell", cells[:5])
def test_cell_faces_point_ids(cell):
    point_ids = {frozenset(cell.get_face(i).point_ids) for i in range(cell.n_faces)}
    assert len(point_ids) == cell.n_faces


@pytest.mark.parametrize("cell", cells[:5], ids=cell_ids[:5])
def test_cell_faces(cell):
    if cell.n_faces:
        assert cell.get_face(0) == cell.faces[0]
        assert cell.get_face(1) != cell.faces[0]
    else:
        with pytest.raises(IndexError, match='Invalid index'):
            cell.get_face(0)


@pytest.mark.parametrize("grid", grids, ids=ids)
def test_cell_bounds(grid):
    assert isinstance(grid.get_cell(0).bounds, tuple)
    assert all(bc >= bg for bc, bg in zip(grid.get_cell(0).bounds[::2], grid.bounds[::2]))
    assert all(bc <= bg for bc, bg in zip(grid.get_cell(0).bounds[1::2], grid.bounds[1::2]))


@pytest.mark.parametrize("cell,type_", zip(cells[:5], types[:5]))
def test_str(cell, type_):
    assert str(type_) in str(cell)


@pytest.mark.parametrize("cell,type_", zip(cells[:5], types[:5]))
def test_repr(cell, type_):
    assert str(type_) in repr(cell)


@pytest.mark.parametrize("cell", cells[:5])
def test_cell_points(cell):
    points = cell.points
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[0] > 0
    assert points.shape[1] == 3


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
