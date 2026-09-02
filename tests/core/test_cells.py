from __future__ import annotations

import gc
import itertools
import re
import sys
from types import GeneratorType

import numpy as np
import pytest
from refleak.testing import Snapshot

import pyvista as pv
from pyvista import Cell
from pyvista import CellType
from pyvista import _vtk
from pyvista.core._vtk_utilities import _SETDATA_TAKES_OWNERSHIP
from pyvista.core._vtk_utilities import _SUPPORTS_FIXED_SIZE_STORAGE
from pyvista.core._vtk_utilities import _SUPPORTS_POLYHEDRON_FACE_CELL_ARRAYS
from pyvista.core.cell import _set_cell_array_data
from pyvista.core.celltype import _CELL_TYPE_INFO
from pyvista.core.celltype import _DEPRECATED_CELL_TYPES
from pyvista.core.celltype import _RENAMED_CELL_TYPES
from pyvista.core.errors import CellSizeError
from pyvista.core.utilities.cells import numpy_to_idarr
from pyvista.examples import cells as example_cells
from pyvista.examples import load_airplane
from pyvista.examples import load_explicit_structured
from pyvista.examples import load_hexbeam
from pyvista.examples import load_rectilinear
from pyvista.examples import load_structured
from pyvista.examples import load_tetbeam
from pyvista.examples import load_uniform
from tests.vtk_backend_divergence import FIXED_SIZE_CELL_STORAGE

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


def test_celltype_dimension_map():
    dimension_map = CellType.dimension_map
    dimensions = list(dimension_map.keys())
    assert dimensions == [0, 1, 2, 3]

    for grouping in dimension_map.values():
        assert isinstance(grouping, frozenset)
        assert all(isinstance(m, CellType) for m in grouping)

    assert CellType.VERTEX in dimension_map[0]
    assert CellType.POLY_VERTEX in dimension_map[0]
    assert CellType.LINE in dimension_map[1]
    assert CellType.POLY_LINE in dimension_map[1]
    assert CellType.TRIANGLE in dimension_map[2]
    assert CellType.QUAD in dimension_map[2]
    assert CellType.POLYGON in dimension_map[2]
    assert CellType.PIXEL in dimension_map[2]
    assert CellType.TRIANGLE_STRIP in dimension_map[2]
    assert CellType.TETRA in dimension_map[3]
    assert CellType.HEXAHEDRON in dimension_map[3]
    assert CellType.VOXEL in dimension_map[3]
    assert CellType.POLYHEDRON in dimension_map[3]

    for i in (0, 1, 2, 3):
        for j in (0, 1, 2, 3):
            if i != j:
                assert dimension_map[i].isdisjoint(dimension_map[j])

    union = dimension_map[0] | dimension_map[1] | dimension_map[2] | dimension_map[3]
    assert union == set(CellType)

    for member in CellType:
        assert member in dimension_map[member.dimension]


def test_celltype_dimension_map_not_mutable():
    mapping = CellType.dimension_map
    match = "'mappingproxy' object does not support item assignment"
    with pytest.raises(TypeError, match=match):
        mapping[42] = 'foo'


def test_abstract_celltype_attributes():
    # ``HIGHER_ORDER_HEXAHEDRON`` has no concrete vtk class, but its dimension
    # is well-defined and the same on every supported VTK build. See
    # https://github.com/pyvista/pyvista/issues/8634
    celltype = pv.CellType.HIGHER_ORDER_HEXAHEDRON
    assert celltype.dimension == 3
    assert not celltype.is_linear

    match = "'HIGHER_ORDER_HEXAHEDRON' without a concrete cell instance."
    with pytest.raises(ValueError, match=match):
        _ = celltype.n_points
    with pytest.raises(ValueError, match=match):
        _ = celltype.n_edges
    with pytest.raises(ValueError, match=match):
        _ = celltype.n_faces


@pytest.mark.parametrize(
    ('celltype', 'expected_dim'),
    [
        ('PARAMETRIC_CURVE', 1),
        ('PARAMETRIC_SURFACE', 2),
        ('PARAMETRIC_TETRA_REGION', 3),
        (pv.CellType.HIGHER_ORDER_CURVE, 1),
        (pv.CellType.HIGHER_ORDER_TRIANGLE, 2),
        (pv.CellType.HIGHER_ORDER_HEXAHEDRON, 3),
        (pv.CellType.LAGRANGE_PYRAMID, 3),
        (pv.CellType.BEZIER_PYRAMID, 3),
    ],
)
def test_abstract_celltype_dimension_is_correct(celltype, expected_dim):
    """Abstract / placeholder cell types report their canonical dimension."""
    if isinstance(celltype, str):
        with pytest.warns(pv.PyVistaDeprecationWarning):
            celltype = getattr(CellType, celltype)

    assert celltype.dimension == expected_dim


@pytest.mark.parametrize('celltype', sorted(_DEPRECATED_CELL_TYPES))
def test_celltype_deprecated(celltype):
    val = _CELL_TYPE_INFO[celltype].value
    match = f'<CellType.{celltype}: {val}> is deprecated and will be removed in a future version.'
    with pytest.warns(pv.PyVistaDeprecationWarning, match=re.escape(match)):
        getattr(CellType, celltype)
    with pytest.warns(pv.PyVistaDeprecationWarning, match=re.escape(match)):
        CellType(val)


@pytest.mark.parametrize('celltype', _RENAMED_CELL_TYPES)
def test_celltype_renamed(celltype):
    val = _CELL_TYPE_INFO[celltype].value
    new_name = _RENAMED_CELL_TYPES[celltype]
    called = CellType(val)
    assert called.name == new_name

    match = f'CellType.{celltype} is deprecated and has been renamed. Use {new_name} instead'
    with pytest.warns(pv.PyVistaDeprecationWarning, match=re.escape(match)):
        getattr(CellType, celltype)


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


@pytest.mark.parametrize('cells', [[-1, 2, 0], [0, -1, 3], [-2, 2, 0]])
def test_init_cell_array_negative_size(cells):
    match = re.escape('Cell array size is invalid. A cell has a negative number of points.')
    with pytest.raises(CellSizeError, match=match):
        _ = pv.core.cell.CellArray(cells)


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
    assert np.array_equal(np.array(connectivity), cell_array.cell_connectivity)
    assert np.array_equal(np.array(offsets), cell_array.cell_offsets)
    assert cell_array.n_cells == cell_array.GetNumberOfCells() == len(offsets) - 1


@pytest.mark.parametrize('deep', [False, True])
def test_init_cell_array_preserves_int32_storage(deep):
    # int32 offsets/connectivity should be stored natively as 32-bit instead of
    # being cast up to int64, which avoids a copy that doubles memory on large
    # meshes. See https://github.com/pyvista/pyvista/issues/8477
    offsets = np.array(OFFSETS_LIST, np.int32)
    connectivity = np.array(CONNECTIVITY_LIST, np.int32)
    cell_array = pv.core.cell.CellArray.from_arrays(offsets, connectivity, deep=deep)
    # The array dtype reflects the native VTK storage width, so an int32 dtype here
    # proves 32-bit storage was kept (no upcast copy). This is checked instead of
    # ``IsStorage32Bit()`` because that method is not available on all supported VTK
    # versions (e.g. 9.4.2).
    assert cell_array.cell_offsets.dtype == np.int32
    assert cell_array.cell_connectivity.dtype == np.int32
    assert np.array_equal(cell_array.cell_offsets, offsets)
    assert np.array_equal(cell_array.cell_connectivity, connectivity)


def test_init_cell_array_int64_uses_64bit_storage():
    # int64 input should keep 64-bit storage (unchanged behavior).
    cell_array = pv.core.cell.CellArray.from_arrays(
        np.array(OFFSETS_LIST, np.int64), np.array(CONNECTIVITY_LIST, np.int64)
    )
    assert cell_array.cell_offsets.dtype == np.int64
    assert cell_array.cell_connectivity.dtype == np.int64


REGULAR_CELL_LIST = [[0, 1, 2], [3, 4, 5]]


@pytest.mark.parametrize(
    'cells',
    [
        REGULAR_CELL_LIST,
        np.array(REGULAR_CELL_LIST, np.int16),
        pytest.param(
            np.array(REGULAR_CELL_LIST, np.int32),
            marks=pytest.mark.xfail(
                sys.platform == 'win32',
                reason='BUG(?) VTK does not use fixed-size storage for int32 cells on Windows',
            ),
        ),
        np.array(REGULAR_CELL_LIST, np.int64),
        np.array(np.vstack(REGULAR_CELL_LIST), order='F'),
    ],
)
@pytest.mark.parametrize('deep', [False, True])
def test_init_cell_array_from_regular_cells(cells, deep):
    cell_array = pv.core.cell.CellArray.from_regular_cells(cells, deep=deep)
    assert np.array_equal(np.array(cells), cell_array.regular_cells)
    assert cell_array.n_cells == cell_array.GetNumberOfCells() == len(cells)
    if _SUPPORTS_FIXED_SIZE_STORAGE:
        assert cell_array.IsStorageFixedSize()


def test_init_cell_array_from_regular_cells_preserves_int32():
    cells = np.array(REGULAR_CELL_LIST, np.int32)
    cell_array = pv.CellArray.from_regular_cells(cells)
    expected_dtype = np.int32 if _SUPPORTS_FIXED_SIZE_STORAGE else pv.ID_TYPE
    assert cell_array.cell_connectivity.dtype == expected_dtype


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


@pytest.mark.parametrize('cell_type_name', _CELL_TYPE_INFO)
def test_cell_types(cell_type_name):
    if not hasattr(_vtk, 'VTK_' + cell_type_name):
        pytest.skip(f'Unsupported cell type {cell_type_name} by VTK')

    if cell_type_name in _DEPRECATED_CELL_TYPES or cell_type_name in _RENAMED_CELL_TYPES:
        with pytest.warns(pv.PyVistaDeprecationWarning):
            pyvista_member = getattr(pv.CellType, cell_type_name)
    else:
        pyvista_member = getattr(pv.CellType, cell_type_name)
    vtk_member = getattr(_vtk, 'VTK_' + cell_type_name)
    assert pyvista_member == vtk_member


def test_n_cells_removed():
    with pytest.raises(TypeError, match=r'unexpected keyword argument'):
        _ = pv.core.cell.CellArray([3, 0, 1, 2], n_cells=1)


@pytest.mark.parametrize('deep', [True, False])
def test_deep_removed(deep: bool):
    with pytest.raises(TypeError, match=r'unexpected keyword argument'):
        _ = pv.core.cell.CellArray([3, 0, 1, 2], deep=deep)


# Fixed-size storage generates the offsets implicitly instead of storing them, so
# `cell_offsets` has to materialize them. That is the case these accessors differ on,
# and it only exists on newer VTK.
OFFSETS_MESH_NAMES = ['PolyData', 'UnstructuredGrid', 'CellArray']
if _SUPPORTS_FIXED_SIZE_STORAGE:
    OFFSETS_MESH_NAMES += ['PolyData (fixed-size)', 'CellArray (fixed-size)']

# `UnstructuredGrid` has no public API that wraps a caller-owned connectivity array
ZERO_COPY_MESH_NAMES = [n for n in OFFSETS_MESH_NAMES if not n.startswith('UnstructuredGrid')]


def _underlying_cell_array(obj):
    if isinstance(obj, pv.PolyData):
        return obj.GetPolys()
    if isinstance(obj, pv.UnstructuredGrid):
        return obj.GetCells()
    return obj


def _offsets_attr(name, array_name):
    """Return the property of ``name`` that holds ``array_name``.

    ``PolyData`` keeps verts, lines, faces and strips in four separate cell arrays, so
    it names them after the cell type. The classes backed by a single cell array name
    them after the cell.
    """
    prefix = 'face' if name.startswith('PolyData') else 'cell'
    return f'{prefix}_{array_name}'


@pytest.fixture
def offsets_meshes(hexbeam):
    """Return one mesh per class that exposes offsets and connectivity properties."""
    meshes = {
        'PolyData': pv.Plane(i_resolution=1, j_resolution=1).triangulate(),
        'UnstructuredGrid': hexbeam,
        'CellArray': pv.CellArray.from_arrays([0, 3, 6], [0, 1, 2, 3, 4, 5]),
    }
    if _SUPPORTS_FIXED_SIZE_STORAGE:
        meshes['PolyData (fixed-size)'] = pv.PolyData.from_regular_faces(
            np.random.default_rng(0).random((4, 3)), [[0, 1, 2], [1, 3, 2]]
        )
        meshes['CellArray (fixed-size)'] = pv.CellArray.from_regular_cells([[0, 1, 2], [3, 4, 5]])
    return meshes


@pytest.mark.skip_vtk_backend('cvista', reason=FIXED_SIZE_CELL_STORAGE)
@pytest.mark.parametrize('name', OFFSETS_MESH_NAMES)
def test_offsets_meshes_storage(offsets_meshes, name):
    # Pin which fixture entries use fixed-size storage so the coverage of the
    # implicit-offsets case cannot silently regress
    if _SUPPORTS_FIXED_SIZE_STORAGE:
        cell_array = _underlying_cell_array(offsets_meshes[name])
        assert cell_array.IsStorageFixedSize() == name.endswith('(fixed-size)')


@pytest.mark.parametrize('name', OFFSETS_MESH_NAMES)
@pytest.mark.parametrize('array_name', ['offsets', 'connectivity'])
def test_offsets_connectivity_is_read_only(offsets_meshes, name, array_name):
    # `UnstructuredGrid.cell_connectivity` predates this API and stays writeable
    # until v0.52, so assert that rather than skipping the case
    expect_writeable = name == 'UnstructuredGrid' and array_name == 'connectivity'
    array = getattr(offsets_meshes[name], _offsets_attr(name, array_name))
    assert array.flags['WRITEABLE'] is expect_writeable
    if not expect_writeable:
        with pytest.raises(ValueError, match='read-only'):
            array[0] = 0


@pytest.mark.parametrize('name', OFFSETS_MESH_NAMES)
def test_offsets_connectivity_describe_cells(offsets_meshes, name):
    obj = offsets_meshes[name]
    offsets = getattr(obj, _offsets_attr(name, 'offsets'))
    connectivity = getattr(obj, _offsets_attr(name, 'connectivity'))
    n_cells = obj.n_faces if name.startswith('PolyData') else obj.n_cells
    assert offsets.size == n_cells + 1
    assert offsets[0] == 0
    assert offsets[-1] == connectivity.size


def test_offsets_setter_polydata():
    mesh = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
    mesh.face_offsets = [0, 2, 4, 6]
    assert mesh.n_faces == 3
    assert np.array_equal(mesh.face_offsets, [0, 2, 4, 6])


def test_connectivity_setter_polydata():
    mesh = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
    mesh.face_connectivity = [2, 1, 0, 2, 3, 1]
    assert np.array_equal(mesh.face_connectivity, [2, 1, 0, 2, 3, 1])
    assert np.array_equal(mesh.regular_faces, [[2, 1, 0], [2, 3, 1]])


POLYDATA_CELL_PREFIXES = ['vert', 'line', 'face', 'strip']


@pytest.fixture
def mixed_polydata():
    """Return a mesh that holds verts, lines, faces and strips at the same time.

    The four cell arrays deliberately hold a different number of cells and different
    point ids, so a property reading the wrong one cannot pass by coincidence.
    """
    mesh = pv.PolyData()
    mesh.points = np.random.default_rng(0).random((8, 3))
    mesh.verts = pv.CellArray.from_arrays([0, 3], [0, 1, 2])
    mesh.lines = pv.CellArray.from_arrays([0, 2, 5], [3, 4, 5, 6, 7])
    mesh.faces = pv.CellArray.from_arrays([0, 3, 6, 10], [0, 1, 2, 2, 3, 4, 4, 5, 6, 7])
    mesh.strips = pv.CellArray.from_arrays([0, 3, 6, 9, 12], [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5])
    return mesh


def test_polydata_offsets_connectivity_are_per_cell_type(mixed_polydata):
    # The faces-only accessors this replaced returned the `faces` arrays whatever the
    # mesh held, so every non-face assertion here fails against that behaviour.
    assert np.array_equal(mixed_polydata.vert_offsets, [0, 3])
    assert np.array_equal(mixed_polydata.vert_connectivity, [0, 1, 2])
    assert np.array_equal(mixed_polydata.line_offsets, [0, 2, 5])
    assert np.array_equal(mixed_polydata.line_connectivity, [3, 4, 5, 6, 7])
    assert np.array_equal(mixed_polydata.face_offsets, [0, 3, 6, 10])
    assert np.array_equal(mixed_polydata.face_connectivity, [0, 1, 2, 2, 3, 4, 4, 5, 6, 7])
    assert np.array_equal(mixed_polydata.strip_offsets, [0, 3, 6, 9, 12])
    assert np.array_equal(mixed_polydata.strip_connectivity, [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5])


@pytest.mark.parametrize(
    ('prefix', 'n_cells'), [('vert', 1), ('line', 2), ('face', 3), ('strip', 4)]
)
def test_polydata_offsets_describe_their_own_cells(mixed_polydata, prefix, n_cells):
    offsets = getattr(mixed_polydata, f'{prefix}_offsets')
    connectivity = getattr(mixed_polydata, f'{prefix}_connectivity')
    assert getattr(mixed_polydata, f'n_{prefix}s') == n_cells
    assert offsets.size - 1 == n_cells
    assert offsets[0] == 0
    assert offsets[-1] == connectivity.size
    # The four counts are all different and none is `n_cells`, which is what makes a
    # single `cell_offsets` pair wrong for this mesh
    assert mixed_polydata.n_cells == 10


@pytest.mark.parametrize('prefix', POLYDATA_CELL_PREFIXES)
@pytest.mark.parametrize('array_name', ['offsets', 'connectivity'])
def test_polydata_offsets_connectivity_is_read_only(mixed_polydata, prefix, array_name):
    array = getattr(mixed_polydata, f'{prefix}_{array_name}')
    assert array.flags['WRITEABLE'] is False
    with pytest.raises(ValueError, match='read-only'):
        array[0] = 0


@pytest.mark.parametrize(
    ('prefix', 'offsets', 'n_cells'),
    [
        ('vert', [0, 1, 3], 2),
        ('line', [0, 5], 1),
        ('face', [0, 5, 10], 2),
        ('strip', [0, 4, 8, 12], 3),
    ],
)
def test_polydata_offsets_setter(mixed_polydata, prefix, offsets, n_cells):
    setattr(mixed_polydata, f'{prefix}_offsets', offsets)
    assert np.array_equal(getattr(mixed_polydata, f'{prefix}_offsets'), offsets)
    assert getattr(mixed_polydata, f'n_{prefix}s') == n_cells


@pytest.mark.parametrize('prefix', POLYDATA_CELL_PREFIXES)
def test_polydata_connectivity_setter_leaves_other_cell_types_alone(mixed_polydata, prefix):
    others = {
        other: (
            getattr(mixed_polydata, f'{other}_offsets').copy(),
            getattr(mixed_polydata, f'{other}_connectivity').copy(),
        )
        for other in POLYDATA_CELL_PREFIXES
        if other != prefix
    }
    reversed_ids = getattr(mixed_polydata, f'{prefix}_connectivity')[::-1].copy()
    setattr(mixed_polydata, f'{prefix}_connectivity', reversed_ids)

    assert np.array_equal(getattr(mixed_polydata, f'{prefix}_connectivity'), reversed_ids)
    for other, (offsets, connectivity) in others.items():
        assert np.array_equal(getattr(mixed_polydata, f'{other}_offsets'), offsets)
        assert np.array_equal(getattr(mixed_polydata, f'{other}_connectivity'), connectivity)


@pytest.mark.parametrize('prefix', ['vert', 'line', 'strip'])
def test_polydata_offsets_connectivity_empty(prefix):
    # A mesh built from faces alone leaves the other three cell arrays empty
    mesh = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
    assert getattr(mesh, f'n_{prefix}s') == 0
    assert np.array_equal(getattr(mesh, f'{prefix}_offsets'), [0])
    assert getattr(mesh, f'{prefix}_connectivity').size == 0


@pytest.mark.parametrize('name', ['cell_offsets', 'cell_connectivity'])
def test_polydata_has_no_cell_offsets_or_connectivity(name):
    # `PolyData` has four cell arrays, so there is no single pair to name `cell_*`
    assert not hasattr(pv.PolyData, name)


def test_connectivity_setter_unstructured_grid(hexbeam):
    connectivity = hexbeam.cell_connectivity.copy()
    connectivity[0] += 1
    hexbeam.cell_connectivity = connectivity
    assert np.array_equal(hexbeam.cell_connectivity, connectivity)
    assert hexbeam.n_cells == hexbeam.celltypes.size
    assert hexbeam.get_cell(0).point_ids[0] == connectivity[0]


@pytest.mark.parametrize('array_name', ['cell_offsets', 'cell_connectivity'])
def test_setter_does_not_alias_input(array_name):
    # Cells must be non-uniform. Uniform offsets take the fixed-size branch, which
    # stores the offsets implicitly and would pass this test without any deep copy.
    cell_array = pv.CellArray.from_arrays([0, 3, 5], [0, 1, 2, 3, 4])
    assert not _SUPPORTS_FIXED_SIZE_STORAGE or not cell_array.IsStorageFixedSize()

    value = getattr(cell_array, array_name).copy()
    setattr(cell_array, array_name, value)
    expected = value.copy()
    value[-1] = 0
    assert np.array_equal(getattr(cell_array, array_name), expected)


def test_unstructured_grid_setter_rejects_cell_count_change(hexbeam):
    connectivity = hexbeam.cell_connectivity
    with pytest.raises(ValueError, match='does not match the number of cell types'):
        hexbeam.cell_offsets = np.arange(0, connectivity.size + 1, 4)


def test_unstructured_grid_setter_rejects_celltype_mismatch():
    # Same number of cells, but the re-partition leaves a TRIANGLE holding 4 points.
    # `celltypes` is carried over unchanged, so this must be rejected.
    grid = pv.UnstructuredGrid(
        np.array([3, 0, 1, 2, 4, 0, 1, 2, 3]),
        np.array([CellType.TRIANGLE, CellType.QUAD], np.uint8),
        np.random.default_rng(0).random((4, 3)),
    )
    with pytest.raises(ValueError, match='its cell type TRIANGLE requires 3'):
        grid.cell_offsets = [0, 4, 7]


def test_unstructured_grid_setter_allows_variable_size_celltype():
    # POLYGON has no fixed point count, so the size check must not apply. Both cells
    # change size, so the check runs on each of them rather than being skipped.
    grid = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 4, 5, 6, 7]),
        np.array([CellType.POLYGON, CellType.POLYGON], np.uint8),
        np.random.default_rng(0).random((8, 3)),
    )
    grid.cell_offsets = [0, 3, 8]
    assert grid.get_cell(0).point_ids == [0, 1, 2]
    assert grid.get_cell(1).point_ids == [3, 4, 5, 6, 7]


def test_unstructured_grid_setter_rejects_polyhedron():
    # A polyhedron is defined by a face stream the cell array does not carry, so
    # replacing the cell array would silently empty the grid.
    face_stream = [4, 3, 0, 1, 2, 3, 0, 1, 3, 3, 0, 2, 3, 3, 1, 2, 3]
    grid = pv.UnstructuredGrid(
        np.array([len(face_stream), *face_stream]),
        np.array([CellType.POLYHEDRON], np.uint8),
        np.random.default_rng(0).random((4, 3)),
    )
    assert grid.n_cells == 1

    with pytest.raises(ValueError, match="Cell type 'POLYHEDRON' cannot be modified"):
        grid.cell_connectivity = grid.cell_connectivity.copy()
    with pytest.raises(ValueError, match="Cell type 'POLYHEDRON' cannot be modified"):
        grid.cell_offsets = grid.cell_offsets.copy()
    assert grid.n_cells == 1


@pytest.mark.parametrize('name', OFFSETS_MESH_NAMES)
@pytest.mark.parametrize(
    ('array_name', 'value', 'error', 'match'),
    [
        ('offsets', [[0, 3], [3, 6]], ValueError, 'must be a 1D array'),
        ('connectivity', [[0, 1], [2, 3]], ValueError, 'must be a 1D array'),
        ('offsets', [0.0, 3.0, 6.0], TypeError, 'integer dtype'),
        ('connectivity', [0.0, 1.0], TypeError, 'integer dtype'),
        ('offsets', [], ValueError, 'at least one value'),
        ('offsets', [1, 4, 6], ValueError, 'first offset must be 0'),
        ('offsets', [0, 6, 3], ValueError, 'monotonically non-decreasing'),
        ('offsets', [0, 3], ValueError, 'must equal the size of the connectivity'),
        ('connectivity', [0, 1, 2], ValueError, 'must equal the size of the connectivity'),
    ],
)
def test_offsets_connectivity_validation(offsets_meshes, name, array_name, value, error, match):
    with pytest.raises(error, match=match):
        setattr(offsets_meshes[name], _offsets_attr(name, array_name), value)


def test_offsets_uses_fixed_size_storage_when_uniform():
    cell_array = pv.CellArray()
    _set_cell_array_data(cell_array, [0, 3, 6], [0, 1, 2, 3, 4, 5])
    if _SUPPORTS_FIXED_SIZE_STORAGE:
        assert cell_array.IsStorageFixedSize()
    assert np.array_equal(cell_array.cell_offsets, [0, 3, 6])


def test_offsets_generic_storage_when_not_uniform():
    cell_array = pv.CellArray()
    _set_cell_array_data(cell_array, [0, 3, 5], [0, 1, 2, 3, 4])
    if _SUPPORTS_FIXED_SIZE_STORAGE:
        assert not cell_array.IsStorageFixedSize()
    assert np.array_equal(cell_array.cell_offsets, [0, 3, 5])


def test_cell_array_offset_array_deprecated():
    cell_array = pv.CellArray.from_arrays([0, 3, 6], [0, 1, 2, 3, 4, 5])
    with pytest.warns(pv.PyVistaDeprecationWarning, match='`CellArray.cell_offsets`'):
        assert np.array_equal(cell_array.offset_array, [0, 3, 6])


def test_cell_array_connectivity_array_deprecated():
    cell_array = pv.CellArray.from_arrays([0, 3, 6], [0, 1, 2, 3, 4, 5])
    with pytest.warns(pv.PyVistaDeprecationWarning, match='`CellArray.cell_connectivity`'):
        assert np.array_equal(cell_array.connectivity_array, [0, 1, 2, 3, 4, 5])


def test_unstructured_grid_offset_deprecated(hexbeam):
    with pytest.warns(pv.PyVistaDeprecationWarning, match='`UnstructuredGrid.cell_offsets`'):
        assert np.array_equal(hexbeam.offset, hexbeam.cell_offsets)


def test_unstructured_grid_connectivity_writeable(hexbeam):
    # `UnstructuredGrid.cell_connectivity` predates this API and returned a writeable
    # array, so it stays writeable through v0.51 rather than breaking callers. The
    # source carries a version guard that fails the build when v0.52 lands.
    assert hexbeam.cell_connectivity.flags['WRITEABLE']
    hexbeam.cell_connectivity[0] += 1
    assert pv.version_info < (0, 52), 'Flip this property to read-only.'


def test_empty_cell_array_offsets_connectivity():
    cell_array = pv.CellArray()
    _set_cell_array_data(cell_array, [0], [])
    assert cell_array.n_cells == 0
    assert np.array_equal(cell_array.cell_offsets, [0])
    assert cell_array.cell_connectivity.size == 0


@pytest.mark.parametrize('name', ZERO_COPY_MESH_NAMES)
def test_documented_zero_copy_cell_array_edit(offsets_meshes, name):
    # The documented zero-copy alternative to assigning a copy to the read-only
    # property: build the cell array with `deep=False` and keep the connectivity array.
    obj = offsets_meshes[name]
    offsets = np.asarray(getattr(obj, _offsets_attr(name, 'offsets')), dtype=pv.ID_TYPE).copy()
    connectivity = np.asarray(
        getattr(obj, _offsets_attr(name, 'connectivity')), dtype=pv.ID_TYPE
    ).copy()
    cell_array = pv.CellArray.from_arrays(offsets, connectivity, deep=False)
    if isinstance(obj, pv.PolyData):
        obj.faces = cell_array
        attr = 'face_connectivity'
    else:
        obj = cell_array
        attr = 'cell_connectivity'
    assert np.shares_memory(connectivity, getattr(obj, attr))

    expected = connectivity[::-1].copy()
    connectivity[:] = expected
    assert np.array_equal(getattr(obj, attr), expected)


POLYHEDRON_ARRAY_NAMES = [
    'polyhedron_face_offsets',
    'polyhedron_face_connectivity',
    'polyhedron_face_location_offsets',
    'polyhedron_face_location_connectivity',
]

requires_polyhedron_face_cell_arrays = pytest.mark.skipif(
    not _SUPPORTS_POLYHEDRON_FACE_CELL_ARRAYS,
    reason='VTK < 9.4 stores a polyhedron as a single padded face stream',
)


@pytest.fixture
def polyhedron_grid():
    """Return a grid holding one polyhedron and one cell that is not a polyhedron.

    Built from literal arrays rather than by merging two example cells. ``merge`` appends
    the main mesh last before VTK 9.5 and first from 9.5 on, so a merged grid orders its
    cells and numbers its points differently depending on the VTK in use.
    """
    # A polyhedron with four triangular faces over points 0-3, then a tetrahedron
    faces = [3, 0, 2, 1, 3, 0, 1, 3, 3, 0, 3, 2, 3, 1, 2, 3]
    cells = [len(faces) + 1, 4, *faces, 4, 4, 5, 6, 7]
    celltypes = [pv.CellType.POLYHEDRON, pv.CellType.TETRA]
    points = np.vstack([example_cells.Polyhedron().points, example_cells.Tetrahedron().points])
    return pv.UnstructuredGrid(cells, celltypes, points)


@requires_polyhedron_face_cell_arrays
def test_polyhedron_face_offsets_connectivity(polyhedron_grid):
    # The polyhedron is a tetrahedron, so it contributes four triangular faces
    offsets = polyhedron_grid.polyhedron_face_offsets
    connectivity = polyhedron_grid.polyhedron_face_connectivity
    assert np.array_equal(offsets, [0, 3, 6, 9, 12])
    assert np.array_equal(connectivity, [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    assert len(offsets) - 1 == 4
    assert offsets[0] == 0
    assert offsets[-1] == connectivity.size


@requires_polyhedron_face_cell_arrays
def test_polyhedron_face_location_offsets_connectivity(polyhedron_grid):
    # Every cell gets an entry, so the cell that is not a polyhedron has an empty slice
    offsets = polyhedron_grid.polyhedron_face_location_offsets
    connectivity = polyhedron_grid.polyhedron_face_location_connectivity
    assert len(offsets) - 1 == polyhedron_grid.n_cells == 2
    assert np.array_equal(offsets, [0, 4, 4])
    assert np.array_equal(connectivity, [0, 1, 2, 3])
    assert offsets[-1] == connectivity.size
    # The same locations the pre-existing padded-stream property reports
    padded = [
        [stop - start, *connectivity[start:stop]] for start, stop in itertools.pairwise(offsets)
    ]
    assert np.array_equal(polyhedron_grid.polyhedron_face_locations, np.concatenate(padded))


@requires_polyhedron_face_cell_arrays
def test_polyhedron_face_locations_index_the_faces(polyhedron_grid):
    offsets = polyhedron_grid.polyhedron_face_offsets
    connectivity = polyhedron_grid.polyhedron_face_connectivity
    faces = [
        connectivity[offsets[face] : offsets[face + 1]].tolist()
        for face in polyhedron_grid.polyhedron_face_location_connectivity
    ]
    assert faces == [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    # The same faces the pre-existing padded-stream property reports
    padded = np.concatenate([[len(face), *face] for face in faces])
    assert np.array_equal(polyhedron_grid.polyhedron_faces, padded)


@requires_polyhedron_face_cell_arrays
@pytest.mark.parametrize('array_name', POLYHEDRON_ARRAY_NAMES)
def test_polyhedron_arrays_are_read_only(polyhedron_grid, array_name):
    array = getattr(polyhedron_grid, array_name)
    assert array.flags['WRITEABLE'] is False
    with pytest.raises(ValueError, match='read-only'):
        array[0] = 0


@requires_polyhedron_face_cell_arrays
@pytest.mark.parametrize('array_name', POLYHEDRON_ARRAY_NAMES)
def test_polyhedron_arrays_have_no_setter(polyhedron_grid, array_name):
    # VTK can only replace the faces together with the cells, cell types and locations
    with pytest.raises(AttributeError):
        setattr(polyhedron_grid, array_name, [0])


@requires_polyhedron_face_cell_arrays
@pytest.mark.parametrize('array_name', POLYHEDRON_ARRAY_NAMES)
def test_polyhedron_arrays_empty_without_a_polyhedron(hexbeam, array_name):
    # VTK allocates neither cell array until the grid holds a polyhedron, so the
    # offsets do not reach `n_cells + 1` here
    expected = [0] if array_name.endswith('_offsets') else []
    assert np.array_equal(getattr(hexbeam, array_name), expected)


@pytest.mark.skipif(
    _SUPPORTS_POLYHEDRON_FACE_CELL_ARRAYS,
    reason='VTK >= 9.4 keeps the polyhedron faces in cell arrays',
)
@pytest.mark.parametrize('array_name', POLYHEDRON_ARRAY_NAMES)
def test_polyhedron_arrays_raise_on_old_vtk(hexbeam, array_name):
    match = re.escape(f'`UnstructuredGrid.{array_name}` requires VTK 9.4 or newer')
    with pytest.raises(pv.VTKVersionError, match=match):
        getattr(hexbeam, array_name)


def test_polydata_connectivity_array_deprecated():
    mesh = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
    with pytest.warns(pv.PyVistaDeprecationWarning, match='`PolyData.face_connectivity`'):
        assert np.array_equal(mesh._connectivity_array, mesh.face_connectivity)


def test_polydata_offset_array_deprecated():
    # Restored after being removed in #8873: downstream code uses these internal
    # helpers, so they get a deprecation cycle rather than being deleted.
    mesh = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
    with pytest.warns(pv.PyVistaDeprecationWarning, match='`PolyData.face_offsets`'):
        assert np.array_equal(mesh._offset_array, mesh.face_offsets)


def test_polydata_offset_array_deprecated_with_fixed_size_storage():
    # The property was removed in #8873 because regular cell arrays store offsets
    # implicitly. Verify it still returns correct values in that case.
    mesh = pv.PolyData.from_regular_faces(
        np.random.default_rng(0).random((4, 3)), [[0, 1, 2], [1, 3, 2]]
    )
    if _SUPPORTS_FIXED_SIZE_STORAGE:
        assert mesh.GetPolys().IsStorageFixedSize()
    with pytest.warns(pv.PyVistaDeprecationWarning, match='`PolyData.face_offsets`'):
        assert np.array_equal(mesh._offset_array, [0, 3, 6])


@pytest.mark.parametrize(
    'make_mesh',
    [
        lambda: pv.PolyData(np.zeros((10, 3))),
        lambda: pv.PolyData.from_regular_faces(np.zeros((4, 3)), [[0, 1, 2], [1, 2, 3]]),
        lambda: pv.lines_from_points(np.zeros((10, 3))),
        lambda: pv.vector_poly_data(np.zeros((10, 3)), np.ones((10, 3))),
    ],
    ids=['verts', 'faces', 'lines', 'vectors'],
)
@pytest.mark.skipif(
    not _SETDATA_TAKES_OWNERSHIP,
    reason='VTK < 9.6 requires CellArray to keep the arrays alive itself',
)
def test_cell_array_does_not_leak_vtk_arrays(make_mesh):
    # A VTK array stashed on the CellArray outlives the mesh in VTK's ghost __dict__
    snapshot = Snapshot(_vtk.vtkObjectBase, label='VTK')
    make_mesh()
    gc.collect()
    snapshot.assert_no_new(when='after building the mesh')


@pytest.mark.parametrize('dtype', [np.int32, pv.ID_TYPE], ids=['int32', 'id_type'])
def test_cell_array_connectivity_outlives_its_source(dtype):
    # The counterpart to the leak test: SetData alone must keep the connectivity valid
    faces = np.arange(120, dtype=dtype).reshape(-1, 3) % 40
    source = pv.PolyData.from_regular_faces(np.zeros((40, 3)), faces)

    shallow = pv.PolyData()
    shallow.shallow_copy(source)
    del source
    gc.collect()
    assert np.array_equal(shallow.regular_faces, faces)

    # Same again, but the CellArray wrapper itself dies while C++ still owns it
    polydata = _vtk.vtkPolyData()
    polydata.SetPoints(pv.vtk_points(np.zeros((40, 3))))
    polydata.SetPolys(pv.CellArray.from_regular_cells(faces.copy()))
    gc.collect()
    assert np.array_equal(pv.wrap(polydata).regular_faces, faces)
