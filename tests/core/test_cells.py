from __future__ import annotations

from types import GeneratorType

import numpy as np
import pytest

import pyvista as pv
from pyvista import Cell
from pyvista import CellType
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.cells import numpy_to_idarr
from pyvista.examples import cells as example_cells
from pyvista.examples import load_airplane
from pyvista.examples import load_explicit_structured
from pyvista.examples import load_hexbeam
from pyvista.examples import load_rectilinear
from pyvista.examples import load_structured
from pyvista.examples import load_tetbeam
from pyvista.examples import load_uniform

grids = [
    load_hexbeam(),
    load_airplane(),
    load_rectilinear(),
    load_structured(),
    load_tetbeam(),
    load_uniform(),
    load_explicit_structured(),
]
ids = [str(type(grid)) for grid in grids]

cells = [
    # 0D cells
    example_cells.Vertex().get_cell(0),
    example_cells.PolyVertex().get_cell(0),
    # 1D cells
    example_cells.Line().get_cell(0),
    example_cells.PolyLine().get_cell(0),
    # 2D cells
    example_cells.Triangle().get_cell(0),
    example_cells.Quadrilateral().get_cell(0),
    example_cells.Polygon().get_cell(0),
    example_cells.TriangleStrip().get_cell(0),
    # 3D cells
    example_cells.Hexahedron().get_cell(0),
    example_cells.Voxel().get_cell(0),
    example_cells.Tetrahedron().get_cell(0),
    example_cells.Polyhedron().get_cell(0),
]
types = [
    # 0D cells
    CellType.VERTEX,
    CellType.POLY_VERTEX,
    # 1D cells
    CellType.LINE,
    CellType.POLY_LINE,
    # 2D cells
    CellType.TRIANGLE,
    CellType.QUAD,
    CellType.POLYGON,
    CellType.TRIANGLE_STRIP,
    # 3D cells
    CellType.HEXAHEDRON,
    CellType.VOXEL,
    CellType.TETRA,
    CellType.POLYHEDRON,
]
dims = [
    # 0D cells
    0,
    0,
    # 1D cells
    1,
    1,
    # 2D cells
    2,
    2,
    2,
    2,
    # 3D cells
    3,
    3,
    3,
    3,
]
npoints = [
    # 0D cells
    1,
    6,
    # 1D cells
    2,
    4,
    # 2D cells
    3,
    4,
    6,
    8,
    # 3D cells
    8,
    8,
    4,
    4,
]
nfaces = [
    # 0D cells
    0,
    0,
    # 1D cells
    0,
    0,
    # 2D cells
    0,
    0,
    0,
    0,
    # 3D cells
    6,
    6,
    4,
    4,
]
nedges = [
    # 0D cells
    0,
    0,
    # 1D cells
    0,
    0,
    # 2D cells
    3,
    4,
    6,
    8,
    # 3D cells
    12,
    12,
    6,
    6,
]
cell_ids = list(map(repr, types))


def test_bad_init():
    with pytest.raises(TypeError, match='must be a vtkCell'):
        _ = Cell(1)


@pytest.mark.parametrize('grid', grids, ids=ids)
def test_cell_attribute(grid):
    assert isinstance(grid.cell, GeneratorType)
    assert all(issubclass(type(cell), Cell) for cell in grid.cell)


@pytest.mark.parametrize('grid', grids, ids=ids)
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
    assert isinstance(hexbeam.get_cell(0), pv.Cell)


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_type_is_inside_enum(cell):
    assert cell.type in CellType


@pytest.mark.parametrize(('cell', 'type_'), zip(cells, types, strict=True), ids=cell_ids)
def test_cell_type(cell, type_):
    assert cell.type == type_


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_is_linear(cell):
    assert cell.is_linear


@pytest.mark.parametrize(('cell', 'dim'), zip(cells, dims, strict=True), ids=cell_ids)
def test_cell_dimension(cell, dim):
    assert cell.dimension == dim


@pytest.mark.parametrize(('cell', 'np'), zip(cells, npoints, strict=True), ids=cell_ids)
def test_cell_n_points(cell, np):
    assert cell.n_points == np


@pytest.mark.parametrize(('cell', 'nf'), zip(cells, nfaces, strict=True), ids=cell_ids)
def test_cell_n_faces(cell, nf):
    assert cell.n_faces == nf


@pytest.mark.parametrize(('cell', 'ne'), zip(cells, nedges, strict=True), ids=cell_ids)
def test_cell_n_edges(cell, ne):
    assert cell.n_edges == ne


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_get_edges(cell):
    assert all(cell.get_edge(i).type == CellType.LINE for i in range(cell.n_edges))

    with pytest.raises(IndexError, match='Invalid index'):
        cell.get_edge(cell.n_edges)


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_edges(cell):
    assert all(edge.type == CellType.LINE for edge in cell.edges)


def test_cell_no_field_data():
    with pytest.raises(NotImplementedError, match='does not support field data'):
        cells[0].add_field_data([1, 2, 3], 'field_data')

    with pytest.raises(NotImplementedError, match='does not support field data'):
        cells[0].clear_field_data()


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_copy_generic(cell):
    cell = cell.copy()
    cell_copy = cell.copy(deep=True)
    assert cell_copy == cell
    cell_copy.points[:] = 1000
    assert cell_copy != cell

    cell_copy = cell.copy(deep=False)
    assert cell_copy == cell
    cell_copy.points[:] = 1000
    assert cell_copy == cell


def test_cell_copy():
    cell = example_cells.Hexahedron().get_cell(0).get_face(0)
    assert isinstance(cell, pv.Cell)
    cell_copy = cell.copy(deep=True)
    assert cell_copy == cell
    cell_copy.points[:] = 0
    assert cell_copy != cell

    cell_copy = cell.copy(deep=False)
    assert cell_copy == cell
    cell_copy.points[:] = 0
    assert cell_copy == cell


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_edges_point_ids(cell):
    point_ids = {frozenset(cell.get_edge(i).point_ids) for i in range(cell.n_edges)}
    assert len(point_ids) == cell.n_edges


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_faces_point_ids(cell):
    point_ids = {frozenset(cell.get_face(i).point_ids) for i in range(cell.n_faces)}
    assert len(point_ids) == cell.n_faces


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_faces(cell):
    if cell.n_faces:
        assert cell.get_face(0) == cell.faces[0]
        assert cell.get_face(1) != cell.faces[0]
    else:
        with pytest.raises(IndexError, match='Invalid index'):
            cell.get_face(0)


@pytest.mark.parametrize('grid', grids, ids=ids)
def test_cell_bounds(grid):
    assert isinstance(grid.get_cell(0).bounds, tuple)
    assert all(
        bc >= bg for bc, bg in zip(grid.get_cell(0).bounds[::2], grid.bounds[::2], strict=True)
    )
    assert all(
        bc <= bg for bc, bg in zip(grid.get_cell(0).bounds[1::2], grid.bounds[1::2], strict=True)
    )


@pytest.mark.parametrize('grid', grids, ids=ids)
def test_cell_center(grid):
    center = grid.get_cell(0).center
    bounds = grid.get_cell(0).bounds

    assert isinstance(center, tuple)
    assert bounds.x_min <= center[0] <= bounds.x_max
    assert bounds.y_min <= center[1] <= bounds.y_max
    assert bounds.z_min <= center[2] <= bounds.z_max


def test_cell_center_value():
    points = [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]]
    cell = [3, 0, 1, 2]
    mesh = pv.PolyData(points, cell)
    assert np.allclose(mesh.get_cell(0).center, [0.5, np.sqrt(3) / 6, 0.0], rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(('cell', 'type_'), zip(cells, types, strict=True), ids=cell_ids)
def test_str(cell, type_):
    assert str(type_) in str(cell)


@pytest.mark.parametrize(('cell', 'type_'), zip(cells, types, strict=True), ids=cell_ids)
def test_repr(cell, type_):
    assert str(type_) in repr(cell)


@pytest.mark.parametrize('cell', cells, ids=cell_ids)
def test_cell_points(cell):
    points = cell.points
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[0] > 0
    assert points.shape[1] == 3


@pytest.mark.parametrize('cell', cells)
def test_cell_cast_to_unstructured_grid(cell):
    grid = cell.cast_to_unstructured_grid()
    assert grid.n_cells == 1
    assert grid.get_cell(0) == cell
    assert grid.get_cell(0).type == cell.type


@pytest.mark.parametrize('cell', cells)
def test_cell_cast_to_polydata(cell):
    if cell.dimension == 3:
        with pytest.raises(
            ValueError,
            match=f'3D cells cannot be cast to PolyData: got cell type {cell.type}',
        ):
            cell.cast_to_polydata()
    else:
        poly = cell.cast_to_polydata()
        assert poly.n_cells == 1
        assert poly.get_cell(0) == cell
        assert poly.get_cell(0).type == cell.type


CELL_LIST = [3, 0, 1, 2, 3, 3, 4, 5]
NCELLS = 2
FCONTIG_ARR = np.array(np.vstack(([3, 0, 1, 2], [3, 3, 4, 5])), order='F')


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
def test_init_cell_array(cells):
    cell_array = pv.core.cell.CellArray(cells)
    assert np.allclose(np.array(cells).ravel(), cell_array.cells)
    assert cell_array.n_cells == cell_array.GetNumberOfCells() == NCELLS


CONNECTIVITY_LIST = [0, 1, 2, 3, 4, 5]
OFFSETS_LIST = [0, 3, 6]


@pytest.mark.parametrize(
    'offsets',
    [
        OFFSETS_LIST,
        np.array(OFFSETS_LIST, np.int16),
        np.array(OFFSETS_LIST, np.int32),
        np.array(OFFSETS_LIST, np.int64),
    ],
)
@pytest.mark.parametrize(
    'connectivity',
    [
        CONNECTIVITY_LIST,
        np.array(CONNECTIVITY_LIST, np.int16),
        np.array(CONNECTIVITY_LIST, np.int32),
        np.array(CONNECTIVITY_LIST, np.int64),
    ],
)
@pytest.mark.parametrize('deep', [False, True])
def test_init_cell_array_from_arrays(offsets, connectivity, deep):
    cell_array = pv.core.cell.CellArray.from_arrays(offsets, connectivity, deep=deep)
    assert np.array_equal(np.array(connectivity), cell_array.connectivity_array)
    assert np.array_equal(np.array(offsets), cell_array.offset_array)
    assert cell_array.n_cells == cell_array.GetNumberOfCells() == len(offsets) - 1


REGULAR_CELL_LIST = [[0, 1, 2], [3, 4, 5]]


@pytest.mark.parametrize(
    'cells',
    [
        REGULAR_CELL_LIST,
        np.array(REGULAR_CELL_LIST, np.int16),
        np.array(REGULAR_CELL_LIST, np.int32),
        np.array(REGULAR_CELL_LIST, np.int64),
        np.array(np.vstack(REGULAR_CELL_LIST), order='F'),
    ],
)
@pytest.mark.parametrize('deep', [False, True])
def test_init_cell_array_from_regular_cells(cells, deep):
    cell_array = pv.core.cell.CellArray.from_regular_cells(cells, deep=deep)
    assert np.array_equal(np.array(cells), cell_array.regular_cells)
    assert cell_array.n_cells == cell_array.GetNumberOfCells() == len(cells)


def test_set_shallow_regular_cells():
    points = [[1.0, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]]
    faces = [[0, 1, 2], [1, 3, 2], [0, 2, 3], [0, 3, 1]]
    meshes = [pv.PolyData.from_regular_faces(points, faces, deep=False) for _ in range(2)]

    for m in meshes:
        assert np.array_equal(m.regular_faces, faces)


def test_numpy_to_idarr_bool():
    mask = np.ones(10, np.bool_)
    idarr = numpy_to_idarr(mask)
    assert np.allclose(mask.nonzero()[0], _vtk.vtk_to_numpy(idarr))


def test_cell_types():
    cell_types = [
        'EMPTY_CELL',
        'VERTEX',
        'POLY_VERTEX',
        'LINE',
        'POLY_LINE',
        'TRIANGLE',
        'TRIANGLE_STRIP',
        'POLYGON',
        'PIXEL',
        'QUAD',
        'TETRA',
        'VOXEL',
        'HEXAHEDRON',
        'WEDGE',
        'PYRAMID',
        'PENTAGONAL_PRISM',
        'HEXAGONAL_PRISM',
        'QUADRATIC_EDGE',
        'QUADRATIC_TRIANGLE',
        'QUADRATIC_QUAD',
        'QUADRATIC_POLYGON',
        'QUADRATIC_TETRA',
        'QUADRATIC_HEXAHEDRON',
        'QUADRATIC_WEDGE',
        'QUADRATIC_PYRAMID',
        'BIQUADRATIC_QUAD',
        'TRIQUADRATIC_HEXAHEDRON',
        'TRIQUADRATIC_PYRAMID',
        'QUADRATIC_LINEAR_QUAD',
        'QUADRATIC_LINEAR_WEDGE',
        'BIQUADRATIC_QUADRATIC_WEDGE',
        'BIQUADRATIC_QUADRATIC_HEXAHEDRON',
        'BIQUADRATIC_TRIANGLE',
        'CUBIC_LINE',
        'CONVEX_POINT_SET',
        'POLYHEDRON',
        'PARAMETRIC_CURVE',
        'PARAMETRIC_SURFACE',
        'PARAMETRIC_TRI_SURFACE',
        'PARAMETRIC_QUAD_SURFACE',
        'PARAMETRIC_TETRA_REGION',
        'PARAMETRIC_HEX_REGION',
        'HIGHER_ORDER_EDGE',
        'HIGHER_ORDER_TRIANGLE',
        'HIGHER_ORDER_QUAD',
        'HIGHER_ORDER_POLYGON',
        'HIGHER_ORDER_TETRAHEDRON',
        'HIGHER_ORDER_WEDGE',
        'HIGHER_ORDER_PYRAMID',
        'HIGHER_ORDER_HEXAHEDRON',
        'LAGRANGE_CURVE',
        'LAGRANGE_TRIANGLE',
        'LAGRANGE_QUADRILATERAL',
        'LAGRANGE_TETRAHEDRON',
        'LAGRANGE_HEXAHEDRON',
        'LAGRANGE_WEDGE',
        'LAGRANGE_PYRAMID',
        'BEZIER_CURVE',
        'BEZIER_TRIANGLE',
        'BEZIER_QUADRILATERAL',
        'BEZIER_TETRAHEDRON',
        'BEZIER_HEXAHEDRON',
        'BEZIER_WEDGE',
        'BEZIER_PYRAMID',
    ]
    for cell_type in cell_types:
        if hasattr(_vtk, 'VTK_' + cell_type):
            assert getattr(pv.CellType, cell_type) == getattr(_vtk, 'VTK_' + cell_type)


def test_n_cells_deprecated():
    with pytest.raises(
        TypeError,
        match=r'CellArray parameter `n_cells` is deprecated and no longer used\.',
    ):
        _ = pv.core.cell.CellArray([3, 0, 1, 2], n_cells=1)
    if pv._version.version_info[:2] > (0, 48):
        msg = 'Remove `n_cells` constructor kwarg'
        raise RuntimeError(msg)


@pytest.mark.parametrize('deep', [True, False])
def test_deep_deprecated(deep: bool):
    with pytest.raises(
        TypeError,
        match=r'CellArray parameter `deep` is deprecated and no longer used\.',
    ):
        _ = pv.core.cell.CellArray([3, 0, 1, 2], deep=deep)
    if pv._version.version_info[:2] > (0, 48):
        msg = 'Remove `deep` constructor kwarg'
        raise RuntimeError(msg)
