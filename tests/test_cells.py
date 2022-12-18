import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import pyvista
from pyvista import Cell, CellType
from pyvista.examples import (
    load_airplane,
    load_explicit_structured,
    load_hexbeam,
    load_rectilinear,
    load_structured,
    load_tetbeam,
    load_uniform,
)


class Test_Cell:

    # Define some lists to test again
    grids = [
        load_hexbeam(),
        load_airplane(),
        load_rectilinear(),
        load_structured(),
        load_tetbeam(),
        load_uniform(),
    ]
    if pyvista._vtk.VTK9:
        grids.append(load_explicit_structured())

    ids = list(map(type, grids))

    types = [
        CellType.HEXAHEDRON,
        CellType.TRIANGLE,
        CellType.VOXEL,
        CellType.QUAD,
        CellType.TETRA,
        CellType.VOXEL,
        CellType.HEXAHEDRON,
    ]

    def test_bad_init(self):
        with pytest.raises(TypeError, match="must be of vtkCell"):
            _ = Cell(1)

    @pytest.mark.parametrize("grid", grids, ids=ids)
    def test_cell_attribute(self, grid):
        assert isinstance(grid.cell, list)
        assert len(grid.cell) == grid.n_cells
        assert all(t == Cell for t in map(type, grid.cell[:]))

    @pytest.mark.parametrize("grid", grids, ids=ids)
    def test_cell_type_is_inside_enum(self, grid):
        assert grid.cell[0].type in CellType

    @pytest.mark.parametrize("grid,type", zip(grids, types), ids=ids)
    def test_cell_type(self, grid, type):
        assert grid.cell[0].type == type

    @pytest.mark.parametrize("grid", grids, ids=ids)
    def test_cell_is_linear(self, grid):
        assert grid.cell[-1].is_linear

    dims = [3, 2, 3, 2, 3, 3, 3]

    @pytest.mark.parametrize("grid, dim", zip(grids, dims), ids=ids)
    def test_cell_dimension(self, grid, dim):
        assert grid.cell[-1].dimension == dim

    npoints = [8, 3, 8, 4, 4, 8, 8]

    @pytest.mark.parametrize("grid, np", zip(grids, npoints), ids=ids)
    def test_cell_n_points(self, grid, np):
        assert grid.cell[-1].n_points == np

    nfaces = [6, 0, 6, 0, 4, 6, 6]

    @pytest.mark.parametrize("grid, nf", zip(grids, nfaces), ids=ids)
    def test_cell_n_faces(self, grid, nf):
        assert grid.cell[-1].n_faces == nf

    nedges = [12, 3, 12, 4, 6, 12, 12]

    @pytest.mark.parametrize("grid, ne", zip(grids, nedges), ids=ids)
    def test_cell_n_edges(self, grid, ne):
        assert grid.cell[-1].n_edges == ne

    @pytest.mark.parametrize("grid", grids, ids=ids)
    def test_cell_edges(self, grid):
        assert isinstance(grid.cell[0].edges, list)
        assert all(e.type == CellType.LINE for e in grid.cell[-1].edges)

    faces_types = [
        CellType.QUAD,
        None,
        CellType.PIXEL,
        None,
        CellType.TRIANGLE,
        CellType.PIXEL,
        CellType.QUAD,
    ]

    @pytest.mark.parametrize("grid, face_type", zip(grids, faces_types), ids=ids)
    def test_cell_faces(self, grid, face_type):
        assert isinstance(grid.cell[0].faces, list)
        if face_type is not None:
            assert all(e.type == face_type for e in grid.cell[-1].faces)
        else:
            assert grid.cell[-1].faces == []

    @pytest.mark.parametrize("grid", grids, ids=ids)
    def test_cell_bounds(self, grid):
        assert isinstance(grid.cell[0].bounds, tuple)
        assert all(bc >= bg for bc, bg in zip(grid.cell[0].bounds[::2], grid.bounds[::2]))
        assert all(bc <= bg for bc, bg in zip(grid.cell[0].bounds[1::2], grid.bounds[1::2]))

    @pytest.mark.parametrize("grid,type", zip(grids, types), ids=ids)
    def test_str(self, grid, type):
        assert str(type) in str(grid.cell[0])

    @pytest.mark.parametrize("grid,type", zip(grids, types), ids=ids)
    def test_repr(self, grid, type):
        assert str(type) in repr(grid.cell[0])

    @pytest.mark.parametrize("grid", grids, ids=ids)
    def test_cell_points(self, grid):
        points = grid.cell[0].points
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
