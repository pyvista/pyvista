from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
from pytest_cases import parametrize

import pyvista as pv
from pyvista import CellType
from pyvista.core import _vtk_core as _vtk
from pyvista.examples import cells
from pyvista.examples.cells import _NOT_SUPPORTED_CELL_SOURCE
from pyvista.examples.cells import _NOT_SUPPORTED_PARAMETRIC

# VTK 9.5.2 swaps Bezier/Lagrange Triangle and Quadrilateral cell types
_VTK_952_SWAPPED_CELLS = frozenset(
    {
        'BezierTriangle',
        'BezierQuadrilateral',
        'LagrangeTriangle',
        'LagrangeQuadrilateral',
    }
)

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
    elif cell_type == pv.CellType.CONVEX_POINT_SET:
        # VTK bug: the convex points are incorrectly flagged as non-convex https://gitlab.kitware.com/vtk/vtk/-/issues/19992
        assert invalid_fields == ('non_convex',)
        # Show that the points are in fact convex as a surface
        assert mesh.extract_surface(algorithm=None).validate_mesh().is_valid

    else:
        assert not invalid_fields


@pytest.mark.needs_vtk_version(9, 6, 0, reason='vtkCellTypeUtilities requires 9.6')
@parametrize('cell_example', cell_example_functions)
def test_cell_name(cell_example):
    # Test that the function names of cell examples match the actual names provided by VTK
    cell_type = next(cell_example().cell).type
    vtk_name = _vtk.vtkCellTypeUtilities.GetTypeAsString(cell_type)
    expected = vtk_name.replace('-', '').replace(' ', '')

    # Special case some cell types
    if expected == 'Polyvertex':
        expected = 'PolyVertex'
    elif expected == 'Polyline':
        expected = 'PolyLine'
    elif expected == 'UnknownCell':
        # VTK bug with these cells https://gitlab.kitware.com/vtk/vtk/-/issues/19988#note_1786036
        if cell_type == CellType.EMPTY_CELL:
            expected = 'Empty'
        if cell_type == CellType.CONVEX_POINT_SET:
            expected = 'ConvexPointSet'

    actual = cell_example.__name__
    assert actual == expected


@pytest.mark.needs_vtk_version(9, 5, 0, reason='VTK bug for higher-order quads/triangles')
@parametrize('cell_example', cell_example_functions)
def test_cell_vtk_class(cell_example):
    if pv.vtk_version_info == (9, 5, 2) and cell_example.__name__ in _VTK_952_SWAPPED_CELLS:
        pytest.skip('VTK 9.5.2 bug: Bezier/Lagrange Triangle and Quadrilateral types are swapped')
    cell = cell_example().GetCell(0)
    celltype = CellType(cell.GetCellType())
    assert celltype.vtk_class is type(cell)


@parametrize('cell_example', cell_example_functions)
def test_cell_dimension(cell_example):
    cell = next(cell_example().cell)
    celltype = CellType(cell.type)
    assert celltype.dimension == cell.dimension


@parametrize('cell_example', cell_example_functions)
def test_cell_is_linear(cell_example):
    cell = next(cell_example().cell)
    celltype = CellType(cell.type)
    assert celltype.is_linear == cell.is_linear


@parametrize('cell_example', cell_example_functions)
def test_cell_is_composite(cell_example):
    cell = next(cell_example().cell)
    celltype = CellType(cell.type)
    assert celltype.is_composite != cell.IsPrimaryCell()


@parametrize('cell_example', cell_example_functions)
def test_cell_n_points(cell_example):
    cell = next(cell_example().cell)
    celltype = CellType(cell.type)
    if (name := cell_example.__name__).startswith(('Bezier', 'Lagrange')) or name in (
        'PolyLine',
        'PolyVertex',
        'Polygon',
        'Polyhedron',
        'QuadraticPolygon',
        'TriangleStrip',
        'ConvexPointSet',
    ):
        match = (
            f'Cannot determine number of points for {celltype.name!r} '
            f'without a concrete cell instance.'
        )
        with pytest.raises(ValueError, match=match):
            _ = celltype.n_points
    else:
        assert celltype.n_points == cell.n_points


@pytest.mark.needs_vtk_version(9, 5, 0, reason='VTK bug for higher-order quads/triangles')
@parametrize('cell_example', cell_example_functions)
def test_cell_n_edges(cell_example):
    if pv.vtk_version_info == (9, 5, 2) and cell_example.__name__ in _VTK_952_SWAPPED_CELLS:
        pytest.skip('VTK 9.5.2 bug: Bezier/Lagrange Triangle and Quadrilateral types are swapped')
    cell = next(cell_example().cell)
    celltype = CellType(cell.type)
    if cell_example.__name__ in (
        'Polygon',
        'Polyhedron',
        'QuadraticPolygon',
        'TriangleStrip',
        'ConvexPointSet',
    ):
        match = (
            f'Cannot determine number of edges for {celltype.name!r} '
            f'without a concrete cell instance.'
        )
        with pytest.raises(ValueError, match=match):
            _ = celltype.n_edges
    else:
        assert celltype.n_edges == cell.n_edges


@parametrize('cell_example', cell_example_functions)
def test_cell_n_faces(cell_example):
    cell = next(cell_example().cell)
    celltype = CellType(cell.type)
    if cell_example.__name__ in ('Polyhedron', 'ConvexPointSet'):
        match = (
            f'Cannot determine number of faces for {celltype.name!r} '
            f'without a concrete cell instance.'
        )
        with pytest.raises(ValueError, match=match):
            _ = celltype.n_faces
    else:
        assert celltype.n_faces == cell.n_faces


def test_abstract_celltype():
    celltype = pv.CellType.HIGHER_ORDER_HEXAHEDRON
    assert not celltype.is_composite
    match = 'Cannot determine number of'
    with pytest.raises(ValueError, match=match):
        _ = celltype.n_points
    with pytest.raises(ValueError, match=match):
        _ = celltype.n_edges
    with pytest.raises(ValueError, match=match):
        _ = celltype.n_faces


@pytest.mark.parametrize('generator', ['examples', 'parametric', 'source'])
@pytest.mark.parametrize('cell_type', [ctype for ctype in CellType if ctype.vtk_class is not None])
def test_generate_cell_blocks(cell_type, generator):
    if pv.vtk_version_info < (9, 4, 0) and generator == 'source':
        pytest.skip('VTK bug with vtkCellTypeSource for some cell types')
    if (generator == 'parametric' and cell_type in _NOT_SUPPORTED_PARAMETRIC) or (
        generator == 'source' and cell_type in _NOT_SUPPORTED_CELL_SOURCE
    ):
        pytest.xfail('Not supported')
    mesh = cells.generate_cell_blocks(cell_type, generator)
    assert isinstance(mesh, pv.MultiBlock)
    assert mesh.n_blocks == 1
    assert mesh.keys() == [cell_type.name]
    assert mesh[0].distinct_cell_types == {cell_type}
    dimensionality = mesh[0].dimensionality
    if dimensionality == 0:
        assert mesh.center == (0.0, 0.0, 0.0)
    elif dimensionality == 1:
        assert mesh.center == (0.5, 0.0, 0.0)
    elif dimensionality == 2:
        assert mesh.center == (0.5, 0.5, 0.0)
    else:
        assert mesh.center == (0.5, 0.5, 0.5)


@pytest.mark.parametrize('generator', ['examples', 'parametric', 'source'])
@pytest.mark.parametrize('dimensions', [(4, 1, 1), (1, 4, 1), (1, 1, 4)])
def test_generate_cell_blocks_block_dimensions(dimensions, generator):
    celltype = CellType.HEXAHEDRON
    mesh = cells.generate_cell_blocks(
        celltype, generator, block_dimensions=dimensions, fill_mode='cycle'
    )
    assert np.allclose(mesh.bounds, (0.0, dimensions[0], 0.0, dimensions[1], 0.0, dimensions[2]))
    assert mesh.keys() == ['HEXAHEDRON', 'HEXAHEDRON_1', 'HEXAHEDRON_2', 'HEXAHEDRON_3']

    # Test blocks are independent copies
    ids = {id(m) for m in mesh}
    assert len(ids) == mesh.n_blocks

    # Test without cycle
    mesh_no_cycle = cells.generate_cell_blocks(
        [celltype] * 4, generator, block_dimensions=dimensions, fill_mode='exact'
    )
    assert mesh == mesh_no_cycle


def test_generate_cell_blocks_block_dimensions_raises():
    match = (
        'Requested dimension (1, 1, 1) is too small. Number of cell types to generate (2) '
        'exceeds the number of blocks requested (1).'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        cells.generate_cell_blocks([CellType.TRIANGLE] * 2, block_dimensions=(1, 1, 1))


@pytest.mark.parametrize('cell_type', _NOT_SUPPORTED_CELL_SOURCE)
def test_generate_cell_blocks_invalid_blocks(cell_type):
    assert cell_type.vtk_class is not None
    match = f"{cell_type!r} is not supported by the 'source' generator."
    with pytest.raises(ValueError, match=match):
        cells.generate_cell_blocks(cell_type, generator='source')


@pytest.mark.parametrize('cell_type', _NOT_SUPPORTED_PARAMETRIC)
def test_generate_cell_blocks_invalid_parametric(cell_type):
    assert cell_type.vtk_class is not None
    match = f"{cell_type!r} is not supported by the 'parametric' generator."
    with pytest.raises(ValueError, match=match):
        cells.generate_cell_blocks(cell_type, generator='parametric')


@pytest.mark.parametrize('generator', ['examples', 'parametric', 'source'])
@pytest.mark.parametrize('cell_type', [ctype for ctype in CellType if ctype.vtk_class is None])
def test_generate_cell_blocks_invalid_abstract(generator, cell_type):
    match = f'{cell_type!r} is not supported'
    with pytest.raises(ValueError, match=match):
        cells.generate_cell_blocks(cell_type, generator=generator)


def test_generate_cell_blocks_unsupported_action():
    match = "-1 is not a valid cell type and is not supported by the 'examples' generator."
    with pytest.raises(ValueError, match=match):
        cells.generate_cell_blocks(-1, unsupported_action='error')
    with pytest.warns(UserWarning, match=match):
        cells.generate_cell_blocks(-1, unsupported_action='warn')

    supported = pv.CellType.HEXAHEDRON
    not_supported = pv.CellType.TRIANGLE_STRIP
    cell_types = [supported, not_supported, supported]
    blocks = cells.generate_cell_blocks(cell_types, unsupported_action='skip', generator='source')
    n_blocks = 3
    assert blocks.n_blocks == n_blocks
    assert blocks[1] is None
    assert blocks.bounds == (0.0, n_blocks, 0.0, 1.0, 0.0, 1.0)

    blocks = cells.generate_cell_blocks(
        cell_types, unsupported_action='squeeze', generator='source'
    )
    n_blocks = 2
    assert blocks.n_blocks == n_blocks
    assert all(isinstance(block, pv.UnstructuredGrid) for block in blocks)
    assert blocks.bounds == (0.0, n_blocks, 0.0, 1.0, 0.0, 1.0)


def test_generate_cell_blocks_fill_mode():
    match = (
        'Requested dimension (2, 1, 1) is too large. Number of cell types to generate (1) '
        'is less than the number of blocks requested (2).\n'
        'Use `fill_mode` to prevent an error from being raised.'
    )
    kwargs = dict(cell_types=CellType.HEXAHEDRON, block_dimensions=(2, 1, 1))
    with pytest.raises(ValueError, match=re.escape(match)):
        cells.generate_cell_blocks(**kwargs)

    mesh = cells.generate_cell_blocks(fill_mode='stop', **kwargs)
    assert mesh.n_blocks == 1
    assert mesh.bounds == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    mesh = cells.generate_cell_blocks(fill_mode='cycle', **kwargs)
    assert mesh.n_blocks == 2
    assert mesh.bounds == (0.0, 2.0, 0.0, 1.0, 0.0, 1.0)


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
