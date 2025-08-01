from __future__ import annotations

import pathlib
from pathlib import Path
import platform
import re
from typing import TYPE_CHECKING
import weakref

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import CellType
from pyvista import examples
from pyvista.core.errors import AmbiguousDataError
from pyvista.core.errors import CellSizeError
from pyvista.core.errors import MissingDataError
from pyvista.examples import cells

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

test_path = str(Path(__file__).resolve().parent)

# must be manually set until pytest adds parametrize with fixture feature
HEXBEAM_CELLS_BOOL = np.ones(40, dtype=bool)  # matches hexbeam.n_cells == 40
STRUCTGRID_CELLS_BOOL = np.ones(729, dtype=bool)  # struct_grid.n_cells == 729
STRUCTGRID_POINTS_BOOL = np.ones(1000, dtype=bool)  # struct_grid.n_points == 1000

pointsetmark = pytest.mark.needs_vtk_version(
    9, 1, 0, reason='Requires VTK>=9.1.0 for a concrete PointSet class'
)


def test_volume(hexbeam):
    assert hexbeam.volume > 0.0


def test_init_from_polydata(sphere):
    unstruct_grid = pv.UnstructuredGrid(sphere)
    assert unstruct_grid.n_points == sphere.n_points
    assert unstruct_grid.n_cells == sphere.n_cells
    assert np.all(unstruct_grid.celltypes == 5)


def test_init_from_structured(struct_grid):
    unstruct_grid = pv.UnstructuredGrid(struct_grid)
    assert unstruct_grid.points.shape[0] == struct_grid.x.size
    assert np.all(unstruct_grid.celltypes == 12)


def test_init_from_unstructured(hexbeam):
    grid = pv.UnstructuredGrid(hexbeam, deep=True)
    grid.points += 1
    assert not np.any(grid.points == hexbeam.points)

    grid = pv.UnstructuredGrid(hexbeam)
    grid.points += 1
    assert np.array_equal(grid.points, hexbeam.points)


def test_init_from_numpy_arrays():
    cells = [[8, 0, 1, 2, 3, 4, 5, 6, 7], [8, 8, 9, 10, 11, 12, 13, 14, 15]]
    cells = np.array(cells).ravel()
    cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON])
    cell1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )

    cell2 = np.array(
        [
            [0, 0, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
            [0, 0, 3],
            [1, 0, 3],
            [1, 1, 3],
            [0, 1, 3],
        ],
        dtype=np.float32,
    )

    points = np.vstack((cell1, cell2))
    grid = pv.UnstructuredGrid(cells, cell_type, points)

    assert grid.number_of_points == 16
    assert grid.number_of_cells == 2


def test_init_bad_input():
    with pytest.raises(TypeError, match='Cannot work with input type'):
        pv.UnstructuredGrid(np.array(1))

    with pytest.raises(TypeError, match='points must have real numbers.'):
        pv.UnstructuredGrid(np.array([2, 0, 1]), np.array(1), 'woa')

    rnd_generator = np.random.default_rng()
    points = rnd_generator.random((4, 3))
    celltypes = [pv.CellType.TETRA]
    cells = np.array([5, 0, 1, 2, 3])
    with pytest.raises(CellSizeError, match='Cell array size is invalid'):
        pv.UnstructuredGrid(cells, celltypes, points)

    with pytest.raises(TypeError, match='requires the following arrays'):
        pv.UnstructuredGrid(*range(5))

    with pytest.raises(TypeError, match='All input types must be sequences.'):
        pv.UnstructuredGrid(*range(3))


def test_check_consistency_raises(mocker: MockerFixture):
    mocker.patch.object(pv.UnstructuredGrid, 'n_cells')
    mocker.patch.object(pv.UnstructuredGrid, 'celltypes')

    grid = pv.UnstructuredGrid()

    with pytest.raises(ValueError):  # noqa: PT011
        grid._check_for_consistency()


def create_hex_example():
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
    cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON], np.int32)

    cell1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )

    cell2 = np.array(
        [
            [0, 0, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
            [0, 0, 3],
            [1, 0, 3],
            [1, 1, 3],
            [0, 1, 3],
        ],
        dtype=np.float32,
    )

    points = np.vstack((cell1, cell2))
    return cells, cell_type, points


def test_init_from_arrays():
    cells, cell_type, points = create_hex_example()
    grid = pv.UnstructuredGrid(cells, cell_type, points, deep=False)

    assert grid.n_cells == 2
    assert np.allclose(cells, grid.cells)
    assert np.allclose(grid.cell_connectivity, np.arange(16))

    # grid.cells is not mutable
    assert not grid.cells.flags['WRITEABLE']

    # but attribute can be set
    new_cells = [8, 0, 1, 2, 3, 4, 5, 6, 7]
    grid.cells = [8, 0, 1, 2, 3, 4, 5, 6, 7]
    assert np.allclose(grid.cells, new_cells)


@pytest.mark.parametrize('multiple_cell_types', [False, True])
@pytest.mark.parametrize('flat_cells', [False, True])
def test_init_from_dict(multiple_cell_types, flat_cells):
    # Try mixed construction
    vtk_cell_format, cell_type, points = create_hex_example()

    offsets = np.array([0, 8, 16])
    cells_hex = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
    input_cells_dict = {CellType.HEXAHEDRON: cells_hex}

    if multiple_cell_types:
        cells_quad = np.array([[16, 17, 18, 19]])

        cell3 = np.array([[0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1]])

        points = np.vstack((points, cell3))
        input_cells_dict[CellType.QUAD] = cells_quad

        # Update expected vtk cell arrays
        vtk_cell_format = np.concatenate([vtk_cell_format, [4], np.squeeze(cells_quad)])
        offsets = np.concatenate([offsets, [20]])
        cell_type = np.concatenate([cell_type, [CellType.QUAD]])

    if flat_cells:
        input_cells_dict = {k: v.reshape([-1]) for k, v in input_cells_dict.items()}

    grid = pv.UnstructuredGrid(input_cells_dict, points, deep=False)

    assert np.all(grid.offset == offsets)
    assert grid.n_cells == (3 if multiple_cell_types else 2)
    assert np.all(grid.cells == vtk_cell_format)
    assert np.allclose(
        grid.cell_connectivity,
        (np.arange(20) if multiple_cell_types else np.arange(16)),
    )

    # Now fetch the arrays
    output_cells_dict = grid.cells_dict

    assert np.all(
        output_cells_dict[CellType.HEXAHEDRON].reshape([-1])
        == input_cells_dict[CellType.HEXAHEDRON].reshape([-1]),
    )

    if multiple_cell_types:
        assert np.all(
            output_cells_dict[CellType.QUAD].reshape([-1])
            == input_cells_dict[CellType.QUAD].reshape([-1]),
        )

    # Test for some errors
    # Invalid index (<0)
    input_cells_dict[CellType.HEXAHEDRON] -= 1

    with pytest.raises(ValueError):  # noqa: PT011
        pv.UnstructuredGrid(input_cells_dict, points, deep=False)

    # Restore
    input_cells_dict[CellType.HEXAHEDRON] += 1

    # Invalid index (>= nr_points)
    input_cells_dict[CellType.HEXAHEDRON].flat[0] = points.shape[0]

    with pytest.raises(ValueError):  # noqa: PT011
        pv.UnstructuredGrid(input_cells_dict, points, deep=False)

    input_cells_dict[CellType.HEXAHEDRON] -= 1

    # Incorrect size
    with pytest.raises(ValueError):  # noqa: PT011
        pv.UnstructuredGrid(
            {CellType.HEXAHEDRON: cells_hex.reshape([-1])[:-1]}, points, deep=False
        )

    # Unknown cell type
    with pytest.raises(ValueError):  # noqa: PT011
        pv.UnstructuredGrid({255: cells_hex}, points, deep=False)

    # Dynamic sizes cell type
    with pytest.raises(ValueError):  # noqa: PT011
        pv.UnstructuredGrid({CellType.POLYGON: cells_hex.reshape([-1])}, points, deep=False)

    # Non-integer arrays
    with pytest.raises(ValueError):  # noqa: PT011
        pv.UnstructuredGrid(
            {CellType.HEXAHEDRON: cells_hex.reshape([-1])[:-1].astype(np.float32)},
            points,
        )

    # Invalid point dimensions
    with pytest.raises(ValueError):  # noqa: PT011
        pv.UnstructuredGrid(input_cells_dict, points[..., :-1])


def test_init_polyhedron():
    polyhedron_nodes = [
        [0.02, 0.0, 0.02],  # 17
        [0.02, 0.01, 0.02],  # 18
        [0.03, 0.01, 0.02],  # 19
        [0.035, 0.005, 0.02],  # 20
        [0.03, 0.0, 0.02],  # 21
        [0.02, 0.0, 0.03],  # 22
        [0.02, 0.01, 0.03],  # 23
        [0.03, 0.01, 0.03],  # 24
        [0.035, 0.005, 0.03],  # 25
        [0.03, 0.0, 0.03],  # 26
    ]
    nodes = np.array(polyhedron_nodes)

    polyhedron_connectivity = [
        3,
        5,
        17,
        18,
        19,
        20,
        21,
        4,
        17,
        18,
        23,
        22,
        4,
        17,
        21,
        26,
        22,
    ]
    cells = np.array([len(polyhedron_connectivity), *polyhedron_connectivity])
    cell_type = np.array([pv.CellType.POLYHEDRON])
    grid = pv.UnstructuredGrid(cells, cell_type, nodes)

    assert grid.n_cells == len(cell_type)
    assert grid.get_cell(0).type == pv.CellType.POLYHEDRON


def test_cells_dict_hexbeam_file():
    grid = pv.UnstructuredGrid(examples.hexbeamfile)
    cells = np.delete(grid.cells, np.arange(0, grid.cells.size, 9)).reshape([-1, 8])

    assert np.all(grid.cells_dict[CellType.HEXAHEDRON] == cells)


def test_cells_dict_variable_length():
    cells_poly = np.concatenate([[5], np.arange(5)])
    cells_types = np.array([CellType.POLYGON])
    points = np.random.default_rng().normal(size=(5, 3))
    grid = pv.UnstructuredGrid(cells_poly, cells_types, points)

    # Dynamic sizes cell types are currently unsupported
    with pytest.raises(ValueError):  # noqa: PT011
        _ = grid.cells_dict

    grid.celltypes[:] = 255
    # Unknown cell types
    with pytest.raises(ValueError):  # noqa: PT011
        _ = grid.cells_dict


def test_cells_dict_empty_grid():
    grid = pv.UnstructuredGrid()
    assert grid.cells_dict == {}


def test_cells_dict_alternating_cells():
    cells = np.concatenate([[4], [1, 2, 3, 4], [3], [0, 1, 2], [4], [0, 1, 5, 6]])
    cells_types = np.array([CellType.QUAD, CellType.TRIANGLE, CellType.QUAD])
    points = np.random.default_rng().normal(size=(3 + 2 * 2, 3))
    grid = pv.UnstructuredGrid(cells, cells_types, points)

    cells_dict = grid.cells_dict

    assert np.all(grid.offset == np.array([0, 4, 7, 11]))
    assert np.all(cells_dict[CellType.QUAD] == np.array([cells[1:5], cells[-4:]]))
    assert np.all(cells_dict[CellType.TRIANGLE] == [0, 1, 2])


def test_destructor():
    ugrid = examples.load_hexbeam()
    ref = weakref.ref(ugrid)
    del ugrid
    assert ref() is None


def test_surface_indices(hexbeam):
    surf = hexbeam.extract_surface()
    surf_ind = surf.point_data['vtkOriginalPointIds']
    assert np.allclose(surf_ind, hexbeam.surface_indices())


def test_extract_feature_edges(hexbeam):
    edges = hexbeam.extract_feature_edges(90, progress_bar=True)
    assert edges.n_points

    edges = hexbeam.extract_feature_edges(180, progress_bar=True)
    assert not edges.n_points


def test_triangulate_inplace(hexbeam):
    hexbeam.triangulate(inplace=True)
    assert (hexbeam.celltypes == CellType.TETRA).all()


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pv.UnstructuredGrid._WRITERS)
def test_save(extension, binary, tmpdir, hexbeam):
    filename = str(tmpdir.mkdir('tmpdir').join(f'tmp.{extension}'))
    if extension == '.vtkhdf' and not binary:
        with pytest.raises(ValueError, match='.vtkhdf files can only be written in binary format'):
            hexbeam.save(filename, binary=binary)
        return

    hexbeam.save(filename, binary=binary)

    grid = pv.UnstructuredGrid(filename)
    assert grid.cells.shape == hexbeam.cells.shape
    assert grid.points.shape == hexbeam.points.shape

    grid = pv.read(filename)
    assert grid.cells.shape == hexbeam.cells.shape
    assert grid.points.shape == hexbeam.points.shape
    assert isinstance(grid, pv.UnstructuredGrid)


def test_pathlib_read_write(tmpdir, hexbeam):
    path = pathlib.Path(str(tmpdir.mkdir('tmpdir').join('tmp.vtk')))
    assert not path.is_file()
    hexbeam.save(path)
    assert path.is_file()

    grid = pv.UnstructuredGrid(path)
    assert grid.cells.shape == hexbeam.cells.shape
    assert grid.points.shape == hexbeam.points.shape

    grid = pv.read(path)
    assert grid.cells.shape == hexbeam.cells.shape
    assert grid.points.shape == hexbeam.points.shape
    assert isinstance(grid, pv.UnstructuredGrid)


def test_init_bad_filename():
    filename = str(Path(test_path) / 'test_grid.py')
    with pytest.raises(IOError):  # noqa: PT011
        pv.UnstructuredGrid(filename)

    with pytest.raises(FileNotFoundError):
        pv.UnstructuredGrid('not a file')


def test_save_bad_extension():
    with pytest.raises(FileNotFoundError):
        pv.UnstructuredGrid('file.abc')


@pytest.mark.parametrize(
    ('nonlinear_input', 'linear_output'),
    [
        (cells.QuadraticQuadrilateral(), cells.Quadrilateral()),
        (cells.QuadraticTriangle(), cells.Triangle()),
        (cells.QuadraticTetrahedron(), cells.Tetrahedron()),
        (cells.QuadraticPyramid(), cells.Pyramid()),
        (cells.QuadraticWedge(), cells.Wedge()),
        (cells.QuadraticHexahedron(), cells.Hexahedron()),
    ],
)
def test_linear_copy(nonlinear_input, linear_output):
    assert not nonlinear_input.get_cell(0).IsLinear()
    lgrid = nonlinear_input.linear_copy()
    assert lgrid.get_cell(0).IsLinear()
    assert lgrid.n_points == nonlinear_input.n_points
    assert lgrid.n_points != linear_output.n_points
    assert lgrid.n_cells == linear_output.n_cells


def test_linear_copy_surf_elem():
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 6, 8, 9, 10, 11, 12, 13], np.int32)
    celltypes = np.array([CellType.QUADRATIC_QUAD, CellType.QUADRATIC_TRIANGLE], np.uint8)

    cell0 = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.1, 0.0],
        [1.1, 0.5, 0.0],
        [0.5, 0.9, 0.0],
        [0.1, 0.5, 0.0],
    ]

    cell1 = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.0, 1.3],
        [0.7, 0.7, 1.3],
        [0.1, 0.1, 1.3],
    ]

    points = np.vstack((cell0, cell1))
    grid = pv.UnstructuredGrid(cells, celltypes, points, deep=False)
    lgrid = grid.linear_copy()

    qfilter = vtk.vtkMeshQuality()
    qfilter.SetInputData(lgrid)
    qfilter.Update()
    qual = pv.wrap(qfilter.GetOutput())['Quality']
    assert np.allclose(qual, [1, 1.4], atol=0.01)


@pytest.mark.parametrize('invert', [True, False])
def test_extract_cells(hexbeam, invert):
    ind = [1, 2, 3]
    n_ind = [i for i in range(hexbeam.n_cells) if i not in ind] if invert else ind

    part_beam = hexbeam.extract_cells(ind, invert=invert)
    assert part_beam.n_cells == len(n_ind)
    assert part_beam.n_points < hexbeam.n_points
    assert np.allclose(part_beam.cell_data['vtkOriginalCellIds'], n_ind)

    mask = np.zeros(hexbeam.n_cells, dtype=bool)
    mask[ind] = True
    part_beam = hexbeam.extract_cells(ind, invert=invert)
    assert part_beam.n_cells == len(n_ind)
    assert part_beam.n_points < hexbeam.n_points
    assert np.allclose(part_beam.cell_data['vtkOriginalCellIds'], n_ind)

    ind = np.vstack(([1, 2], [4, 5]))[:, 0]
    part_beam = hexbeam.extract_cells(ind)


def test_merge(hexbeam):
    grid = hexbeam.copy()
    grid.points[:, 0] += 1
    unmerged = grid.merge(hexbeam, inplace=False, merge_points=False)

    grid.merge(hexbeam, inplace=True, merge_points=True)
    assert grid.n_points > hexbeam.n_points
    assert grid.n_points < unmerged.n_points


@pytest.mark.needs_vtk_version(
    less_than=(9, 5, 0), reason='Main always has priority for vtk >= 9.5.'
)
def test_merge_not_main(hexbeam):
    grid = hexbeam.copy()
    grid.points[:, 0] += 1
    with pytest.warns(
        pv.PyVistaDeprecationWarning, match=r"The keyword 'main_has_priority' is deprecated"
    ):
        unmerged = grid.merge(hexbeam, inplace=False, merge_points=False, main_has_priority=False)

    grid.merge(hexbeam, inplace=True, merge_points=True)
    assert grid.n_points > hexbeam.n_points
    assert grid.n_points < unmerged.n_points


def test_merge_order():
    key = 'data'
    main = examples.cells.Quadrilateral()
    main_array = [0, 0, 0, 0]
    main.point_data[key] = main_array
    main_celltype = main.celltypes[0]

    other = examples.cells.Pixel()
    other_array = [1, 1, 1, 1]
    other.point_data[key] = other_array
    other_celltype = other.celltypes[0]

    merged = main.merge(other)
    expected_array = main_array
    actual_array = merged.point_data[key]
    assert np.array_equal(actual_array, expected_array)

    if pv.vtk_version_info >= (9, 5, 0):
        expected_celltypes = [main_celltype, other_celltype]
    else:
        expected_celltypes = [other_celltype, main_celltype]
    actual_celltypes = merged.celltypes
    assert np.array_equal(actual_celltypes, expected_celltypes)


def test_merge_list(hexbeam):
    grid_a = hexbeam.copy()
    grid_a.points[:, 0] += 1

    grid_b = hexbeam.copy()
    grid_b.points[:, 1] += 1

    grid_a.merge([hexbeam, grid_b], inplace=True, merge_points=True)
    assert grid_a.n_points > hexbeam.n_points


def test_merge_invalid(hexbeam, sphere):
    with pytest.raises(TypeError):
        sphere.merge([hexbeam], inplace=True)


def test_init_structured_raise():
    with pytest.raises(TypeError, match='Invalid parameters'):
        pv.StructuredGrid(['a', 'b', 'c'])
    with pytest.raises(ValueError, match='Too many args'):
        pv.StructuredGrid([0, 1], [0, 1], [0, 1], [0, 1])


def test_init_structured(struct_grid):
    xrng = np.arange(-10, 10, 2, dtype=np.float32)
    yrng = np.arange(-10, 10, 2, dtype=np.float32)
    zrng = np.arange(-10, 10, 2, dtype=np.float32)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    grid = pv.StructuredGrid(x, y, z)
    assert np.allclose(struct_grid.x, x)
    assert np.allclose(struct_grid.y, y)
    assert np.allclose(struct_grid.z, z)

    grid_a = pv.StructuredGrid(grid, deep=True)
    grid_a.points += 1
    assert not np.any(grid_a.points == grid.points)

    grid_a = pv.StructuredGrid(grid)
    grid_a.points += 1
    assert np.array_equal(grid_a.points, grid.points)


@pytest.fixture
def structured_points():
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    source = np.empty((x.size, 3), x.dtype)
    source[:, 0] = x.ravel('F')
    source[:, 1] = y.ravel('F')
    source[:, 2] = z.ravel('F')
    return source, (*x.shape, 1)


def test_no_copy_polydata_init():
    source = np.random.default_rng().random((100, 3))
    mesh = pv.PolyData(source)
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_polydata_points_setter():
    source = np.random.default_rng().random((100, 3))
    mesh = pv.PolyData()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_structured_mesh_init(structured_points):
    source, dims = structured_points
    mesh = pv.StructuredGrid(source)
    mesh.dimensions = dims
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_structured_mesh_points_setter(structured_points):
    source, dims = structured_points
    mesh = pv.StructuredGrid()
    mesh.points = source
    mesh.dimensions = dims
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


@pointsetmark
def test_no_copy_pointset_init():
    source = np.random.default_rng().random((100, 3))
    mesh = pv.PointSet(source)
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


@pointsetmark
def test_no_copy_pointset_points_setter():
    source = np.random.default_rng().random((100, 3))
    mesh = pv.PointSet()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_unstructured_grid_points_setter():
    source = np.random.default_rng().random((100, 3))
    mesh = pv.UnstructuredGrid()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_rectilinear_grid():
    xrng = np.arange(-10, 10, 2, dtype=float)
    yrng = np.arange(-10, 10, 5, dtype=float)
    zrng = np.arange(-10, 10, 1, dtype=float)
    mesh = pv.RectilinearGrid(xrng, yrng, zrng)
    x = mesh.x
    x /= 2
    assert np.array_equal(mesh.x, x)
    assert np.may_share_memory(mesh.x, x)
    assert np.array_equal(mesh.x, xrng)
    assert np.may_share_memory(mesh.x, xrng)
    y = mesh.y
    y /= 2
    assert np.array_equal(mesh.y, y)
    assert np.may_share_memory(mesh.y, y)
    assert np.array_equal(mesh.y, yrng)
    assert np.may_share_memory(mesh.y, yrng)
    z = mesh.z
    z /= 2
    assert np.array_equal(mesh.z, z)
    assert np.may_share_memory(mesh.z, z)
    assert np.array_equal(mesh.z, zrng)
    assert np.may_share_memory(mesh.z, zrng)


def test_grid_repr(struct_grid):
    str_ = str(struct_grid)
    assert 'StructuredGrid' in str_
    assert f'N Points:     {struct_grid.n_points}\n' in str_

    repr_ = repr(struct_grid)
    assert 'StructuredGrid' in repr_
    assert f'N Points:     {struct_grid.n_points}\n' in repr_


def test_slice_structured(struct_grid):
    sliced = struct_grid[1, :, 1:3]  # three different kinds of slices
    assert sliced.dimensions == (1, struct_grid.dimensions[1], 2)

    # check that points are in the right place
    assert struct_grid.x[1, :, 1:3].ravel() == pytest.approx(sliced.x.ravel())
    assert struct_grid.y[1, :, 1:3].ravel() == pytest.approx(sliced.y.ravel())
    assert struct_grid.z[1, :, 1:3].ravel() == pytest.approx(sliced.z.ravel())

    with pytest.raises(TypeError):
        # fancy indexing error
        struct_grid[[1, 2, 3], :, 1:3]

    with pytest.raises(RuntimeError):
        # incorrect number of dims error
        struct_grid[:, :]


def test_invalid_init_structured():
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 2)
    zrng = np.arange(-10, 10, 2)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    z = z[:, :, :2]
    with pytest.raises(ValueError):  # noqa: PT011
        pv.StructuredGrid(x, y, z)


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pv.StructuredGrid._WRITERS)
def test_save_structured(extension, binary, tmpdir, struct_grid):
    filename = str(tmpdir.mkdir('tmpdir').join(f'tmp.{extension}'))
    struct_grid.save(filename, binary=binary)

    grid = pv.StructuredGrid(filename)
    assert grid.x.shape == struct_grid.y.shape
    assert grid.n_cells
    assert grid.points.shape == struct_grid.points.shape

    grid = pv.read(filename)
    assert grid.x.shape == struct_grid.y.shape
    assert grid.n_cells
    assert grid.points.shape == struct_grid.points.shape
    assert isinstance(grid, pv.StructuredGrid)


def test_load_structured_bad_filename():
    with pytest.raises(FileNotFoundError):
        pv.StructuredGrid('not a file')

    filename = str(Path(test_path) / 'test_grid.py')
    with pytest.raises(IOError):  # noqa: PT011
        pv.StructuredGrid(filename)


def test_instantiate_by_filename():
    ex = examples

    # actual mapping of example file to datatype
    fname_to_right_type = {
        ex.antfile: pv.PolyData,
        ex.planefile: pv.PolyData,
        ex.hexbeamfile: pv.UnstructuredGrid,
        ex.spherefile: pv.PolyData,
        ex.uniformfile: pv.ImageData,
        ex.rectfile: pv.RectilinearGrid,
    }

    # a few combinations of wrong type
    fname_to_wrong_type = {
        ex.antfile: pv.UnstructuredGrid,  # actual data is PolyData
        ex.planefile: pv.StructuredGrid,  # actual data is PolyData
        ex.rectfile: pv.UnstructuredGrid,  # actual data is StructuredGrid
    }

    # load the files into the right types
    for fname, right_type in fname_to_right_type.items():
        data = right_type(fname)
        assert data.n_points > 0

    # load the files into the wrong types
    for fname, wrong_type in fname_to_wrong_type.items():
        with pytest.raises(TypeError):
            data = wrong_type(fname)


def test_create_rectilinear_grid_from_specs():
    # 3D example
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 5)
    zrng = np.arange(-10, 10, 1)
    grid = pv.RectilinearGrid(xrng)
    assert grid.n_cells == 9
    assert grid.n_points == 10
    grid = pv.RectilinearGrid(xrng, yrng)
    assert grid.n_cells == 9 * 3
    assert grid.n_points == 10 * 4
    grid = pv.RectilinearGrid(xrng, yrng, zrng)
    assert grid.n_cells == 9 * 3 * 19
    assert grid.n_points == 10 * 4 * 20
    assert grid.bounds == (-10.0, 8.0, -10.0, 5.0, -10.0, 9.0)

    # with Sequence
    xrng = [0, 1]
    yrng = [0, 1, 2]
    zrng = [0, 1, 2, 3]
    grid = pv.RectilinearGrid(xrng)
    assert grid.n_cells == 1
    assert grid.n_points == 2
    grid = pv.RectilinearGrid(xrng, yrng)
    assert grid.n_cells == 2
    assert grid.n_points == 6
    grid = pv.RectilinearGrid(xrng, yrng, zrng)
    assert grid.n_cells == 6
    assert grid.n_points == 24

    # 2D example
    cell_spacings = np.array([1.0, 1.0, 2.0, 2.0, 5.0, 10.0])
    x_coordinates = np.cumsum(cell_spacings)
    y_coordinates = np.cumsum(cell_spacings)
    grid = pv.RectilinearGrid(x_coordinates, y_coordinates)
    assert grid.n_cells == 5 * 5
    assert grid.n_points == 6 * 6
    assert grid.bounds == (1.0, 21.0, 1.0, 21.0, 0.0, 0.0)


def test_create_rectilinear_after_init():
    x = np.array([0, 1, 2])
    y = np.array([0, 5, 8])
    z = np.array([3, 2, 1])
    grid = pv.RectilinearGrid()
    grid.x = x
    assert grid.dimensions == (3, 1, 1)
    grid.y = y
    assert grid.dimensions == (3, 3, 1)
    grid.z = z
    assert grid.dimensions == (3, 3, 3)
    assert np.allclose(grid.x, x)
    assert np.allclose(grid.y, y)
    assert np.allclose(grid.z, z)


def test_create_rectilinear_grid_from_file():
    grid = examples.load_rectilinear()
    assert grid.n_cells == 16146
    assert grid.n_points == 18144
    assert grid.bounds == (-350.0, 1350.0, -400.0, 1350.0, -850.0, 0.0)
    assert grid.n_arrays == 1


def test_read_rectilinear_grid_from_file():
    grid = pv.read(examples.rectfile)
    assert grid.n_cells == 16146
    assert grid.n_points == 18144
    assert grid.bounds == (-350.0, 1350.0, -400.0, 1350.0, -850.0, 0.0)
    assert grid.n_arrays == 1


def test_read_rectilinear_grid_from_pathlib():
    grid = pv.RectilinearGrid(pathlib.Path(examples.rectfile))
    assert grid.n_cells == 16146
    assert grid.n_points == 18144
    assert grid.bounds == (-350.0, 1350.0, -400.0, 1350.0, -850.0, 0.0)
    assert grid.n_arrays == 1


def test_raise_rectilinear_grid_non_unique():
    rng_uniq = np.arange(4.0)
    rng_dupe = np.array([0, 1, 2, 2], dtype=float)
    with pytest.raises(ValueError, match='Array contains duplicate values'):
        pv.RectilinearGrid(rng_dupe, check_duplicates=True)
    with pytest.raises(ValueError, match='Array contains duplicate values'):
        pv.RectilinearGrid(rng_uniq, rng_dupe, check_duplicates=True)
    with pytest.raises(ValueError, match='Array contains duplicate values'):
        pv.RectilinearGrid(rng_uniq, rng_uniq, rng_dupe, check_duplicates=True)


def test_cast_rectilinear_grid():
    grid = pv.read(examples.rectfile)
    structured = grid.cast_to_structured_grid()
    assert isinstance(structured, pv.StructuredGrid)
    assert structured.n_points == grid.n_points
    assert structured.n_cells == grid.n_cells
    assert np.allclose(structured.points, grid.points)
    for k, v in grid.point_data.items():
        assert np.allclose(structured.point_data[k], v)
    for k, v in grid.cell_data.items():
        assert np.allclose(structured.cell_data[k], v)


def test_create_image_data_from_specs():
    # empty
    grid = pv.ImageData()

    # create ImageData
    dims = (10, 10, 10)
    grid = pv.ImageData(dimensions=dims)  # Using default spacing and origin
    assert grid.dimensions == dims
    assert grid.extent == (0, 9, 0, 9, 0, 9)
    assert grid.origin == (0.0, 0.0, 0.0)
    assert grid.spacing == (1.0, 1.0, 1.0)

    # Using default origin
    spacing = (2, 1, 5)
    grid = pv.ImageData(dimensions=dims, spacing=spacing)
    assert grid.dimensions == dims
    assert grid.origin == (0.0, 0.0, 0.0)
    assert grid.spacing == spacing
    origin = (10, 35, 50)

    # Everything is specified
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
    assert grid.dimensions == dims
    assert grid.origin == origin
    assert grid.spacing == spacing

    # ensure negative spacing is not allowed
    match = 'spacing values must all be greater than or equal to 0.0.'
    with pytest.raises(ValueError, match=match):
        grid = pv.ImageData(dimensions=dims, spacing=(-1, 1, 1))

    # uniform grid from a uniform grid
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
    grid_from_grid = pv.ImageData(grid)
    assert grid == grid_from_grid

    # and is a copy
    grid.origin = (0, 0, 0)
    assert grid != grid_from_grid


def test_image_data_init_kwargs():
    vector = (1, 2, 3)
    image = pv.ImageData(dimensions=vector)
    assert image.dimensions == vector

    image = pv.ImageData(spacing=vector)
    assert image.spacing == vector

    image = pv.ImageData(origin=vector)
    assert image.origin == vector

    matrix = np.eye(3) * 2
    image = pv.ImageData(direction_matrix=matrix)
    assert np.allclose(image.direction_matrix, matrix)

    image = pv.ImageData(offset=vector)
    assert np.allclose(image.offset, vector)


@pytest.mark.parametrize('dims', [None, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_image_data_empty_init(dims):
    image = pv.ImageData(dimensions=dims)
    assert image.n_points == 0
    assert image.n_cells == 0
    assert image.area == 0
    assert image.volume == 0

    points = image.points
    assert np.array_equal(points, np.zeros((0, 3)))


def test_image_data_invald_args():
    with pytest.raises(TypeError):
        pv.ImageData(1)


def test_uniform_setters():
    grid = pv.ImageData()
    grid.dimensions = (10, 10, 10)
    assert grid.GetDimensions() == (10, 10, 10)
    assert grid.dimensions == (10, 10, 10)
    grid.spacing = (5, 2, 1)
    assert grid.GetSpacing() == (5, 2, 1)
    assert grid.spacing == (5, 2, 1)
    grid.origin = (6, 27.7, 19.8)
    assert grid.GetOrigin() == (6, 27.7, 19.8)
    assert grid.origin == (6, 27.7, 19.8)


def test_create_image_data_from_file():
    grid = examples.load_uniform()
    assert grid.n_cells == 729
    assert grid.n_points == 1000
    assert grid.bounds == (0.0, 9.0, 0.0, 9.0, 0.0, 9.0)
    assert grid.n_arrays == 2
    assert grid.dimensions == (10, 10, 10)


def test_read_image_data_from_file():
    grid = pv.read(examples.uniformfile)
    assert grid.n_cells == 729
    assert grid.n_points == 1000
    assert grid.bounds == (0.0, 9.0, 0.0, 9.0, 0.0, 9.0)
    assert grid.n_arrays == 2
    assert grid.dimensions == (10, 10, 10)


def test_read_image_data_from_pathlib():
    grid = pv.ImageData(pathlib.Path(examples.uniformfile))
    assert grid.n_cells == 729
    assert grid.n_points == 1000
    assert grid.bounds == (0.0, 9.0, 0.0, 9.0, 0.0, 9.0)
    assert grid.n_arrays == 2
    assert grid.dimensions == (10, 10, 10)


def test_cast_uniform_to_structured():
    grid = examples.load_uniform()
    structured = grid.cast_to_structured_grid()
    assert structured.n_points == grid.n_points
    assert structured.n_arrays == grid.n_arrays
    assert structured.bounds == grid.bounds


def test_cast_uniform_to_rectilinear():
    grid = examples.load_uniform()
    grid.offset = (1, 2, 3)
    grid.direction_matrix = np.diag((-1.0, 1.0, 1.0))
    grid.spacing = (1.1, 2.2, 3.3)
    rectilinear = grid.cast_to_rectilinear_grid()
    assert rectilinear.n_points == grid.n_points
    assert rectilinear.n_arrays == grid.n_arrays
    assert rectilinear.bounds == grid.bounds

    grid.direction_matrix = pv.Transform().rotate_x(30).matrix[:3, :3]
    match = (
        'The direction matrix is not a diagonal matrix and cannot be used when casting to '
        'RectilinearGrid.\nThe direction is ignored. Consider casting to StructuredGrid instead.'
    )
    with pytest.warns(RuntimeWarning, match=match):
        rectilinear = grid.cast_to_rectilinear_grid()
    # Input has orientation, output does not
    assert rectilinear.bounds != grid.bounds
    # Test output has orientation component removed
    grid.direction_matrix = np.eye(3)
    assert rectilinear.bounds == grid.bounds


def test_cast_image_data_with_float_spacing_to_rectilinear():
    # https://github.com/pyvista/pyvista/pull/6656
    grid = pv.ImageData(
        dimensions=(10, 10, 10),
        spacing=(27.88888888888889, 28.11111111111111, 28.22222222222222),
        origin=(-126.0, -127.0, -127.0),
    )
    rectilinear = grid.cast_to_rectilinear_grid()
    assert rectilinear.n_points == grid.n_points
    assert rectilinear.n_arrays == grid.n_arrays
    assert rectilinear.bounds == grid.bounds


def test_image_data_to_tetrahedra():
    grid = pv.ImageData(dimensions=(2, 2, 2))
    ugrid = grid.to_tetrahedra()
    assert ugrid.n_cells == 5


def test_fft_and_rfft(noise_2d):
    grid = pv.ImageData(dimensions=(10, 10, 1))
    with pytest.raises(MissingDataError, match='FFT filter requires point scalars'):
        grid.fft()

    grid['cell_data'] = np.arange(grid.n_cells)
    with pytest.raises(MissingDataError, match='FFT filter requires point scalars'):
        grid.fft()

    name = noise_2d.active_scalars_name
    noise_fft = noise_2d.fft()
    assert noise_fft[name].dtype == np.complex128

    full_pass = noise_2d.fft().rfft()
    assert full_pass[name].dtype == np.complex128

    # expect FFT and and RFFT to transform from time --> freq --> time domain
    assert np.allclose(noise_2d['scalars'], full_pass[name].real)
    assert np.allclose(full_pass[name].imag, 0)

    output_scalars_name = 'out_scalars'
    # also, disable active scalars to check if it will be automatically set
    noise_2d.active_scalars_name = None
    noise_fft = noise_2d.fft(output_scalars_name=output_scalars_name)
    assert output_scalars_name in noise_fft.point_data

    noise_fft = noise_2d.fft()
    noise_fft_inactive_scalars = noise_fft.copy()
    noise_fft_inactive_scalars.active_scalars_name = None
    full_pass = noise_fft_inactive_scalars.rfft()
    assert np.allclose(full_pass.active_scalars, noise_fft.rfft().active_scalars)


def test_fft_low_pass(noise_2d):
    name = noise_2d.active_scalars_name
    noise_no_scalars = noise_2d.copy()
    noise_no_scalars.clear_data()
    with pytest.raises(MissingDataError, match='FFT filters require point scalars'):
        noise_no_scalars.low_pass(1, 1, 1)

    noise_too_many_scalars = noise_no_scalars.copy()
    noise_too_many_scalars.point_data.set_array(np.arange(noise_2d.n_points), 'a')
    noise_too_many_scalars.point_data.set_array(np.arange(noise_2d.n_points), 'b')
    with pytest.raises(AmbiguousDataError, match='There are multiple point scalars available'):
        noise_too_many_scalars.low_pass(1, 1, 1)

    with pytest.raises(ValueError, match='must be complex data'):
        noise_2d.low_pass(1, 1, 1)

    out_zeros = noise_2d.fft().low_pass(0, 0, 0)
    assert np.allclose(out_zeros[name][1:], 0)

    out = noise_2d.fft().low_pass(1, 1, 1)
    assert not np.allclose(out[name][1:], 0)


def test_fft_high_pass(noise_2d):
    name = noise_2d.active_scalars_name
    out_zeros = noise_2d.fft().high_pass(100000, 100000, 100000)
    assert np.allclose(out_zeros[name], 0)

    out = noise_2d.fft().high_pass(10, 10, 10)
    assert not np.allclose(out[name], 0)


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['.vtk', '.vtr'])
def test_save_rectilinear(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir('tmpdir').join(f'tmp.{extension}'))
    ogrid = examples.load_rectilinear()
    ogrid.save(filename, binary=binary)
    grid = pv.RectilinearGrid(filename)
    assert grid.n_cells == ogrid.n_cells
    assert np.allclose(grid.x, ogrid.x)
    assert np.allclose(grid.y, ogrid.y)
    assert np.allclose(grid.z, ogrid.z)
    assert grid.dimensions == ogrid.dimensions
    grid = pv.read(filename)
    assert isinstance(grid, pv.RectilinearGrid)
    assert grid.n_cells == ogrid.n_cells
    assert np.allclose(grid.x, ogrid.x)
    assert np.allclose(grid.y, ogrid.y)
    assert np.allclose(grid.z, ogrid.z)
    assert grid.dimensions == ogrid.dimensions


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['.vtk', '.vti'])
@pytest.mark.parametrize('reader', [pv.ImageData, pv.read])
@pytest.mark.parametrize('direction_matrix', [np.eye(3), np.diag((-1, 1, -1))])
def test_save_uniform(extension, binary, tmpdir, uniform, reader, direction_matrix):
    filename = str(tmpdir.mkdir('tmpdir').join(f'tmp{extension}'))
    is_identity_matrix = np.allclose(direction_matrix, np.eye(3))
    uniform.direction_matrix = direction_matrix

    if extension == '.vtk' and not is_identity_matrix:
        match = re.escape(
            'The direction matrix for ImageData will not be saved using the legacy `.vtk` format.'
            '\nSee https://gitlab.kitware.com/vtk/vtk/-/issues/19663 '
            '\nUse the `.vti` extension instead (XML format).'
        )
        with pytest.warns(UserWarning, match=match):
            uniform.save(filename, binary=binary)
    else:
        uniform.save(filename, binary=binary)

    grid = reader(filename)

    if extension == '.vtk' and not is_identity_matrix:
        # Direction matrix is lost
        assert not np.allclose(grid.direction_matrix, uniform.direction_matrix)
        # Add it back manually for equality check
        grid.direction_matrix = uniform.direction_matrix

    assert grid == uniform


def test_grid_points():
    """Test the points methods on ImageData and RectilinearGrid"""
    # test creation of 2d grids
    x = y = range(3)
    z = [0]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]
    grid = pv.ImageData()
    with pytest.raises(AttributeError):
        grid.points = points
    grid.origin = (0.0, 0.0, 0.0)
    grid.dimensions = (3, 3, 1)
    grid.spacing = (1, 1, 1)
    assert grid.n_points == 9
    assert grid.n_cells == 4
    assert np.allclose(grid.points, points)

    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
    )
    grid = pv.ImageData()
    grid.dimensions = [2, 2, 2]
    grid.spacing = [1, 1, 1]
    grid.origin = [0.0, 0.0, 0.0]
    assert np.allclose(np.unique(grid.points, axis=0), np.unique(points, axis=0))
    opts = np.c_[grid.x, grid.y, grid.z]
    assert np.allclose(np.unique(opts, axis=0), np.unique(points, axis=0))

    # Now test rectilinear grid
    grid = pv.RectilinearGrid()
    with pytest.raises(AttributeError):
        grid.points = points
    x, y, z = np.array([0, 1, 3]), np.array([0, 2.5, 5]), np.array([0, 1])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid.x = x
    grid.y = y
    grid.z = z
    assert grid.dimensions == (3, 3, 2)
    assert np.allclose(grid.meshgrid, (xx, yy, zz))
    assert np.allclose(
        grid.points,
        np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')],
    )


def test_imagedata_direction_matrix():
    # Create image data with a single voxel cell
    image = pv.ImageData(dimensions=(2, 2, 2))
    assert image.n_points == 8
    assert image.n_cells == 1

    initial_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    assert image.bounds == initial_bounds

    # Test set/get
    expected_matrix = pv.Transform().rotate_vector((1, 2, 3), 30).matrix[:3, :3]
    image.direction_matrix = expected_matrix
    assert np.array_equal(image.direction_matrix, expected_matrix)

    # Test bounds using a transformed reference box
    box = pv.Box(bounds=initial_bounds)
    box.transform(image.index_to_physical_matrix, inplace=True)
    expected_bounds = box.bounds
    assert np.allclose(image.bounds, expected_bounds)

    # Check that filters make use of the direction matrix internally
    image['data'] = np.ones((image.n_points,))
    filtered = image.threshold()
    assert filtered.bounds == expected_bounds

    # Check that points make use of the direction matrix internally
    poly_points = pv.PolyData(image.points)
    assert np.allclose(poly_points.bounds, expected_bounds)


def test_imagedata_direction_matrix_orthonormal(uniform):
    # Test matrix does not enforce orthogonality
    matrix_not_orthonormal = np.reshape(range(1, 10), (3, 3))
    uniform.direction_matrix = matrix_not_orthonormal
    assert np.array_equal(uniform.direction_matrix, matrix_not_orthonormal)


def test_imagedata_index_to_physical_matrix():
    # Create image with arbitrary translation (origin) and rotation (direction)
    image = pv.ImageData()
    vector = (1, 2, 3)
    rotation = pv.Transform().rotate_vector(vector, 30).matrix[:3, :3]
    image.origin = vector
    image.direction_matrix = rotation

    expected_transform = pv.Transform().rotate(rotation).translate(vector)
    ijk_to_xyz = image.index_to_physical_matrix
    assert np.allclose(ijk_to_xyz, expected_transform.matrix)

    xyz_to_ijk = image.physical_to_index_matrix
    assert np.allclose(xyz_to_ijk, expected_transform.inverse_matrix)

    # Test setters
    I3 = np.eye(3)
    I4 = np.eye(4)
    image.index_to_physical_matrix = I4
    assert np.allclose(image.index_to_physical_matrix, I4)
    assert np.allclose(image.spacing, (1, 1, 1))
    assert np.allclose(image.origin, (0, 0, 0))
    assert np.allclose(image.direction_matrix, I3)

    image.physical_to_index_matrix = expected_transform.inverse_matrix
    xyz_to_ijk = image.physical_to_index_matrix
    assert np.allclose(xyz_to_ijk, expected_transform.inverse_matrix)


def test_grid_extract_selection_points(struct_grid):
    grid = pv.UnstructuredGrid(struct_grid)
    sub_grid = grid.extract_points([0])
    assert sub_grid.n_cells == 1

    sub_grid = grid.extract_points(range(100))
    assert sub_grid.n_cells > 1


def test_gaussian_smooth():
    uniform = examples.load_uniform()
    active = uniform.active_scalars_name
    values = uniform.active_scalars

    uniform = uniform.gaussian_smooth(scalars=active)
    assert uniform.active_scalars_name == active
    assert uniform.active_scalars.shape == values.shape
    assert not np.all(uniform.active_scalars == values)
    values = uniform.active_scalars

    uniform = uniform.gaussian_smooth(radius_factor=5, std_dev=1.3)
    assert uniform.active_scalars_name == active
    assert uniform.active_scalars.shape == values.shape
    assert not np.all(uniform.active_scalars == values)


@pytest.mark.parametrize('ind', [range(10), np.arange(10), HEXBEAM_CELLS_BOOL])
def test_remove_cells(ind, hexbeam, request):
    if pv.vtk_version_info < (9, 4, 0) and platform.system() == 'Linux':
        request.node.expect_vtk_error = True
    grid_copy = hexbeam.remove_cells(ind)
    assert grid_copy.n_cells < hexbeam.n_cells


@pytest.mark.parametrize('ind', [range(10), np.arange(10), HEXBEAM_CELLS_BOOL])
def test_remove_cells_not_inplace(ind, hexbeam):
    grid_copy = hexbeam.copy()  # copy to protect
    grid_w_removed = grid_copy.remove_cells(ind)
    assert grid_w_removed.n_cells < hexbeam.n_cells
    assert grid_copy.n_cells == hexbeam.n_cells


def test_remove_cells_invalid(hexbeam):
    grid_copy = hexbeam.copy()
    with pytest.raises(ValueError):  # noqa: PT011
        grid_copy.remove_cells(np.ones(10, dtype=bool), inplace=True)


@pytest.mark.parametrize('ind', [range(10), np.arange(10), STRUCTGRID_CELLS_BOOL])
def test_hide_cells(ind, struct_grid):
    struct_grid.hide_cells(ind, inplace=True)
    assert struct_grid.HasAnyBlankCells()

    out = struct_grid.hide_cells(ind, inplace=False)
    assert id(out) != id(struct_grid)
    assert out.HasAnyBlankCells()

    with pytest.raises(ValueError, match='Boolean array size must match'):
        struct_grid.hide_cells(np.ones(10, dtype=bool), inplace=True)


@pytest.mark.parametrize('ind', [range(10), np.arange(10), STRUCTGRID_POINTS_BOOL])
def test_hide_points(ind, struct_grid):
    struct_grid.hide_points(ind)
    assert struct_grid.HasAnyBlankPoints()

    with pytest.raises(ValueError, match='Boolean array size must match'):
        struct_grid.hide_points(np.ones(10, dtype=bool))


def test_set_extent():
    uni_grid = pv.ImageData(dimensions=[10, 10, 10])
    with pytest.raises(ValueError):  # noqa: PT011
        uni_grid.extent = [0, 1]

    extent = [0, 1, 0, 1, 0, 1]
    uni_grid.extent = extent
    assert np.array_equal(uni_grid.extent, extent)


def test_set_extent_width_spacing():
    grid = pv.ImageData(
        dimensions=(10, 10, 10),
        origin=(-0.5, -0.3, -0.1),
        spacing=(0.1, 0.05, 0.01),
    )
    grid.extent = (5, 9, 0, 9, 0, 9)
    assert np.allclose(grid.x[:5], [0.0, 0.1, 0.2, 0.3, 0.4])


def test_imagedata_offset():
    grid = pv.ImageData()
    offset = (1, 2, 3)
    grid.extent = (offset[0], 9, offset[1], 9, offset[2], 9)
    actual_dimensions = grid.dimensions
    actual_offset = grid.offset
    assert isinstance(actual_offset, tuple)
    assert actual_offset == offset
    # Test to make sure dimensions are unchanged since setting offset
    # modifies the extent which could modify dimensions.
    assert grid.dimensions == actual_dimensions


def test_unstructured_grid_cast_to_explicit_structured_grid():
    grid = examples.load_explicit_structured()
    grid = grid.hide_cells(range(80, 120))
    grid = grid.cast_to_unstructured_grid()
    grid = grid.cast_to_explicit_structured_grid()
    assert grid.n_cells == 120
    assert grid.n_points == 210
    assert grid.bounds == (0.0, 80.0, 0.0, 50.0, 0.0, 6.0)
    assert 'BLOCK_I' in grid.cell_data
    assert 'BLOCK_J' in grid.cell_data
    assert 'BLOCK_K' in grid.cell_data
    assert 'vtkGhostType' in grid.cell_data
    assert np.count_nonzero(grid.cell_data['vtkGhostType']) == 40


def test_unstructured_grid_cast_to_explicit_structured_grid_raises():
    with pytest.raises(
        TypeError,
        match="'BLOCK_I', 'BLOCK_J' and 'BLOCK_K' cell arrays are required",
    ):
        pv.UnstructuredGrid().cast_to_explicit_structured_grid()


def test_explicit_structured_grid_init():
    grid = examples.load_explicit_structured()
    assert isinstance(grid, pv.ExplicitStructuredGrid)
    assert grid.n_cells == 120
    assert grid.n_points == 210
    assert grid.bounds == (0.0, 80.0, 0.0, 50.0, 0.0, 6.0)
    assert repr(grid) == str(grid)
    assert 'N Cells' in str(grid)
    assert 'N Points' in str(grid)
    assert 'N Arrays' in str(grid)

    dims = (2, 2, 3)
    cells = {pv.CellType.HEXAHEDRON: np.arange(16).reshape(2, 8)}
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [1.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ]
    grid = pv.ExplicitStructuredGrid(dims, cells, points)
    assert grid.n_cells == 2
    assert grid.n_points == 16


def test_explicit_structured_grid_cast_to_unstructured_grid():
    block_i = np.fromstring(
        """
        0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
        1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1
        2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2
        3 0 1 2 3 0 1 2 3
        """,
        sep=' ',
        dtype=int,
    )

    block_j = np.fromstring(
        """
        0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4
        4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3
        3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2
        2 3 3 3 3 4 4 4 4
        """,
        sep=' ',
        dtype=int,
    )

    block_k = np.fromstring(
        """
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3
        3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5
        5 5 5 5 5 5 5 5 5
        """,
        sep=' ',
        dtype=int,
    )

    grid = examples.load_explicit_structured()
    grid = grid.cast_to_unstructured_grid()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert 'BLOCK_I' in grid.cell_data
    assert 'BLOCK_J' in grid.cell_data
    assert 'BLOCK_K' in grid.cell_data
    assert np.array_equal(grid.cell_data['BLOCK_I'], block_i)
    assert np.array_equal(grid.cell_data['BLOCK_J'], block_j)
    assert np.array_equal(grid.cell_data['BLOCK_K'], block_k)


def test_explicit_structured_grid_save():
    grid = examples.load_explicit_structured()
    grid = grid.hide_cells(range(80, 120))
    grid.save('grid.vtu')
    grid = pv.ExplicitStructuredGrid('grid.vtu')
    assert grid.n_cells == 120
    assert grid.n_points == 210
    assert grid.bounds == (0.0, 80.0, 0.0, 50.0, 0.0, 6.0)
    assert np.count_nonzero(grid.cell_data['vtkGhostType']) == 40
    Path('grid.vtu').unlink()


def test_explicit_structured_grid_save_raises():
    with pytest.raises(ValueError, match='Cannot save texture of a pointset.'):
        examples.load_explicit_structured().save('test.vtu', texture=np.array([]))


def test_explicit_structured_grid_hide_cells():
    ghost = np.asarray(
        """
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32
    32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32
    """.split(),  # noqa: SIM905
        dtype=np.uint8,
    )

    grid = examples.load_explicit_structured()

    copy = grid.hide_cells(range(80, 120))
    assert isinstance(copy, pv.ExplicitStructuredGrid)
    assert 'vtkGhostType' in copy.cell_data
    assert 'vtkGhostType' not in grid.cell_data
    assert np.array_equal(copy.cell_data['vtkGhostType'], ghost)

    out = grid.hide_cells(range(80, 120), inplace=True)
    assert out is grid
    assert 'vtkGhostType' in grid.cell_data
    assert np.array_equal(grid.cell_data['vtkGhostType'], ghost)


def test_explicit_structured_grid_show_cells():
    grid = examples.load_explicit_structured()
    grid.hide_cells(range(80, 120), inplace=True)

    copy = grid.show_cells()
    assert isinstance(copy, pv.ExplicitStructuredGrid)
    assert 'vtkGhostType' in copy.cell_data
    assert np.count_nonzero(copy.cell_data['vtkGhostType']) == 0
    assert np.count_nonzero(grid.cell_data['vtkGhostType']) == 40

    out = grid.show_cells(inplace=True)
    assert out is grid
    assert np.count_nonzero(grid.cell_data['vtkGhostType']) == 0


def test_explicit_structured_grid_dimensions():
    grid = examples.load_explicit_structured()
    assert isinstance(grid.dimensions, tuple)
    assert isinstance(grid.dimensions[0], int)
    assert len(grid.dimensions) == 3
    assert grid.dimensions == (5, 6, 7)


def test_explicit_structured_grid_visible_bounds():
    grid = examples.load_explicit_structured()
    grid = grid.hide_cells(range(80, 120))
    assert isinstance(grid.visible_bounds, tuple)
    assert all(isinstance(x, float) for x in grid.visible_bounds)
    assert len(grid.visible_bounds) == 6
    assert grid.visible_bounds == (0.0, 80.0, 0.0, 50.0, 0.0, 4.0)


def test_explicit_structured_grid_cell_id():
    grid = examples.load_explicit_structured()

    ind = grid.cell_id((3, 4, 0))
    assert np.issubdtype(ind, np.integer)
    assert ind == 19

    ind = grid.cell_id([(3, 4, 0), (3, 2, 1), (1, 0, 2), (2, 3, 2)])
    assert isinstance(ind, np.ndarray)
    assert np.issubdtype(ind.dtype, np.integer)
    assert np.array_equal(ind, [19, 31, 41, 54])


def test_explicit_structured_grid_cell_coords():
    grid = examples.load_explicit_structured()

    coords = grid.cell_coords(19)
    assert isinstance(coords, np.ndarray)
    assert np.issubdtype(coords.dtype, np.integer)
    assert np.array_equal(coords, (3, 4, 0))

    coords = grid.cell_coords((19, 31, 41, 54))
    assert isinstance(coords, np.ndarray)
    assert np.issubdtype(coords.dtype, np.integer)
    assert np.array_equal(coords, [(3, 4, 0), (3, 2, 1), (1, 0, 2), (2, 3, 2)])


def test_explicit_structured_grid_neighbors():
    grid = examples.load_explicit_structured()

    with pytest.raises(ValueError, match='Invalid value for `rel`'):
        indices = grid.neighbors(0, rel='foo')

    indices = grid.neighbors(0, rel='topological')
    assert isinstance(indices, list)
    assert all(np.issubdtype(ind, np.integer) for ind in indices)
    assert indices == [1, 4, 20]

    indices = grid.neighbors(0, rel='connectivity')
    assert isinstance(indices, list)
    assert all(np.issubdtype(ind, np.integer) for ind in indices)
    assert indices == [1, 4, 20]

    indices = grid.neighbors(0, rel='geometric')
    assert isinstance(indices, list)
    assert all(np.issubdtype(ind, np.integer) for ind in indices)
    assert indices == [1, 4, 20]


def test_explicit_structured_grid_compute_connectivity():
    connectivity = np.asarray(
        """
    42 43 43 41 46 47 47 45 46 47 47 45 46 47 47 45 38 39 39 37 58 59 59 57
    62 63 63 61 62 63 63 61 62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61
    62 63 63 61 62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61 62 63 63 61
    62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61 62 63 63 61 62 63 63 61
    54 55 55 53 26 27 27 25 30 31 31 29 30 31 31 29 30 31 31 29 22 23 23 21
    """.split(),  # noqa: SIM905
        dtype=int,
    )

    grid = examples.load_explicit_structured()
    assert 'ConnectivityFlags' not in grid.cell_data

    copy = grid.compute_connectivity()
    assert isinstance(copy, pv.ExplicitStructuredGrid)
    assert 'ConnectivityFlags' in copy.cell_data
    assert 'ConnectivityFlags' not in grid.cell_data
    assert np.array_equal(copy.cell_data['ConnectivityFlags'], connectivity)

    out = grid.compute_connectivity(inplace=True)
    assert out is grid
    assert 'ConnectivityFlags' in grid.cell_data
    assert np.array_equal(grid.cell_data['ConnectivityFlags'], connectivity)


def test_explicit_structured_grid_compute_connections():
    connections = np.asarray(
        """
    3 4 4 3 4 5 5 4 4 5 5 4 4 5 5 4 3 4 4 3 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4
    5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4 5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6
    6 5 4 5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4 5 5 4 3 4 4 3 4 5 5 4 4 5 5
    4 4 5 5 4 3 4 4 3
    """.split(),  # noqa: SIM905
        dtype=int,
    )

    grid = examples.load_explicit_structured()
    assert 'number_of_connections' not in grid.cell_data

    copy = grid.compute_connections()
    assert isinstance(copy, pv.ExplicitStructuredGrid)
    assert 'number_of_connections' in copy.cell_data
    assert 'number_of_connections' not in grid.cell_data
    assert np.array_equal(copy.cell_data['number_of_connections'], connections)

    grid.compute_connections(inplace=True)
    assert 'number_of_connections' in grid.cell_data
    assert np.array_equal(grid.cell_data['number_of_connections'], connections)


def test_explicit_structured_grid_raise_init():
    with pytest.raises(ValueError, match='Too many args'):
        pv.ExplicitStructuredGrid(1, 2, 3, True)

    with pytest.raises(ValueError, match='Expected dimensions to be length 3'):
        pv.ExplicitStructuredGrid((1, 2), np.random.default_rng().random((4, 3)))

    with pytest.raises(ValueError, match='Expected dimensions to be length 3'):
        pv.ExplicitStructuredGrid(
            (1, 2),
            np.random.default_rng().integers(10, size=9),
            np.random.default_rng().random((8, 3)),
        )

    with pytest.raises(ValueError, match='Expected cells to be length 54'):
        pv.ExplicitStructuredGrid(
            (2, 3, 4),
            np.random.default_rng().integers(10, size=9 * 6 - 1),
            np.random.default_rng().random((8, 3)),
        )

    with pytest.raises(ValueError, match='Expected cells to be a single cell of type 12'):
        pv.ExplicitStructuredGrid(
            (2, 3, 4),
            {CellType.QUAD: np.random.default_rng().integers(10, size=(10, 8))},
            np.random.default_rng().random((8, 3)),
        )

    with pytest.raises(ValueError, match='Expected cells to be of shape'):
        pv.ExplicitStructuredGrid(
            (2, 3, 4),
            {CellType.HEXAHEDRON: np.random.default_rng().integers(10, size=(10, 8))},
            np.random.default_rng().random((8, 3)),
        )


@pytest.mark.needs_vtk_version(
    9, 2, 2, reason='Requires VTK>=9.2.2 for ExplicitStructuredGrid.clean'
)
def test_explicit_structured_grid_clean():
    grid = examples.load_explicit_structured()

    # Duplicate points
    ugrid = grid.cast_to_unstructured_grid().copy()
    cells = ugrid.cells.reshape((ugrid.n_cells, 9))[:, 1:]
    ugrid.cells = np.column_stack(
        (
            np.full(ugrid.n_cells, 8),
            np.arange(8 * ugrid.n_cells).reshape((ugrid.n_cells, 8)),
        )
    ).ravel()
    ugrid.points = np.concatenate(ugrid.points[cells])
    assert ugrid.n_points == 960

    egrid = ugrid.cast_to_explicit_structured_grid().clean()
    assert egrid.n_points == grid.n_points


@pointsetmark
def test_structured_grid_cast_to_explicit_structured_grid():
    grid = examples.download_office()
    grid = grid.hide_cells(np.arange(80, 120))
    grid = pv.ExplicitStructuredGrid(grid)
    assert grid.n_cells == 7220
    assert grid.n_points == 8400
    assert 'vtkGhostType' in grid.cell_data
    assert (grid.cell_data['vtkGhostType'] > 0).sum() == 40


def test_structured_grid_cast_to_explicit_structured_grid_raises():
    xrng = np.arange(-10, 10, 20, dtype=np.float32)
    x, y, z = np.meshgrid(*[xrng] * 3, indexing='ij')
    grid = pv.StructuredGrid(x, y, z)
    with pytest.raises(
        TypeError,
        match='Only 3D structured grid can be casted to an explicit structured grid.',
    ):
        grid.cast_to_explicit_structured_grid()


def test_copy_no_copy_wrap_object(datasets):
    for dataset in datasets:
        # different dataset types have different copy behavior for points
        # use point data which is common
        dataset['data'] = np.ones(dataset.n_points)
        new_dataset = type(dataset)(dataset)
        new_dataset['data'] += 1
        assert np.array_equal(new_dataset['data'], dataset['data'])

    for dataset in datasets:
        # different dataset types have different copy behavior for points
        # use point data which is common
        dataset['data'] = np.ones(dataset.n_points)
        new_dataset = type(dataset)(dataset, deep=True)
        new_dataset['data'] += 1
        assert not np.any(new_dataset['data'] == dataset['data'])


def test_copy_no_copy_wrap_object_vtk9(datasets_vtk9):
    for dataset in datasets_vtk9:
        # different dataset types have different copy behavior for points
        # use point data which is common
        dataset['data'] = np.ones(dataset.n_points)
        new_dataset = type(dataset)(dataset)
        new_dataset['data'] += 1
        assert np.array_equal(new_dataset['data'], dataset['data'])

    for dataset in datasets_vtk9:
        # different dataset types have different copy behavior for points
        # use point data which is common
        dataset['data'] = np.ones(dataset.n_points)
        new_dataset = type(dataset)(dataset, deep=True)
        new_dataset['data'] += 1
        assert not np.any(new_dataset['data'] == dataset['data'])


@pytest.mark.parametrize('grid_class', [pv.RectilinearGrid, pv.ImageData])
@pytest.mark.parametrize(
    ('dimensionality', 'dimensions'),
    [(0, (1, 1, 1)), (1, (1, 42, 1)), (2, (42, 1, 142)), (3, (2, 42, 142))],
)
def test_grid_dimensionality(grid_class, dimensionality, dimensions):
    if grid_class == pv.ImageData:
        grid = grid_class(dimensions=dimensions)
    elif grid_class == pv.RectilinearGrid:
        grid = grid_class(range(dimensions[0]), range(dimensions[1]), range(dimensions[2]))

    assert grid.dimensionality == dimensionality
    assert grid.dimensionality == grid.get_cell(0).GetCellDimension()


@pytest.mark.parametrize('arg', [1, True, object()])
def test_rect_grid_raises(arg):
    with pytest.raises(
        TypeError,
        match=re.escape(f'Type ({type(arg)}) not understood by `RectilinearGrid`'),
    ):
        pv.RectilinearGrid(arg)


@given(args=st.lists(st.none()).filter(lambda x: len(x) in [2, 3]))
def test_rect_grid_raises_args(args):
    with pytest.raises(
        TypeError,
        match=re.escape('Arguments not understood by `RectilinearGrid`.'),
    ):
        pv.RectilinearGrid(*args)


def test_rect_grid_dimensions_raises():
    g = pv.RectilinearGrid()
    match = re.escape(
        'The dimensions of a `RectilinearGrid` are implicitly defined and thus cannot be set.',
    )
    with pytest.raises(AttributeError, match=match):
        g.dimensions = 1


@pytest.fixture
def empty_poly_cast_to_ugrid():
    cast_ugrid = pv.PolyData().cast_to_unstructured_grid()

    # Likely VTK bug, these should not be None but they are
    assert cast_ugrid.GetCells() is None
    assert cast_ugrid.GetCellTypesArray() is None

    # Make sure a proper ugrid does not have these as None
    ugrid = pv.UnstructuredGrid()
    assert isinstance(ugrid.GetCells(), vtk.vtkCellArray)
    assert isinstance(ugrid.GetCellTypesArray(), vtk.vtkUnsignedCharArray)

    return cast_ugrid


def test_cells_empty(empty_poly_cast_to_ugrid):
    assert empty_poly_cast_to_ugrid.cells.size == 0


def test_celltypes_empty(empty_poly_cast_to_ugrid, hexbeam):
    celltypes = empty_poly_cast_to_ugrid.celltypes
    assert celltypes.size == 0
    assert celltypes.dtype == hexbeam.celltypes.dtype


def test_cell_connectivity_empty(empty_poly_cast_to_ugrid, hexbeam):
    connectivity = empty_poly_cast_to_ugrid.cell_connectivity
    assert connectivity.size == 0
    assert connectivity.dtype == hexbeam.cell_connectivity.dtype


@pytest.fixture
def appended_images():
    def create_slice(ind: 0):
        im = pv.ImageData(dimensions=(3, 3, 1), offset=(0, 0, ind))
        im.point_data['data'] = np.ones((im.n_points,)) * ind
        return im

    slice0 = create_slice(0)
    slice1 = create_slice(1)

    append = vtk.vtkImageAppend()
    append.SetAppendAxis(2)
    append.AddInputData(slice0)
    append.AddInputData(slice1)
    append.Update()
    return slice0, slice1, pv.wrap(append.GetOutput())


@pytest.fixture
def appended_images_with_offset(appended_images):
    offset = (1, 2, 3)
    slice0, slice1, appended = appended_images
    slice0.offset = slice0.offset + np.array(offset)
    slice1.offset = slice1.offset + np.array(offset)
    appended.offset = appended.offset + np.array(offset)
    return slice0, slice1, appended


def test_imagedata_slice_index_with_slice(uniform):
    sliced = uniform.slice_index(slice(10), slice(0, 10), slice(None))
    assert sliced == uniform


def test_imagedata_slice_index_strict_index(uniform):
    rng = [None, uniform.dimensions[0] + 1]
    uniform.slice_index(rng)  # No error
    match = (
        'The requested volume of interest (0, 10, 0, 9, 0, 9) '
        "is outside the input's extent (0, 9, 0, 9, 0, 9)."
    )
    with pytest.raises(IndexError, match=re.escape(match)):
        uniform.slice_index(rng, strict_index=True)

    rng = [-uniform.dimensions[0] - 1, None]
    uniform.slice_index(rng)  # No error
    match = (
        'The requested volume of interest (-1, 9, 0, 9, 0, 9) '
        "is outside the input's extent (0, 9, 0, 9, 0, 9)."
    )
    with pytest.raises(IndexError, match=re.escape(match)):
        uniform.slice_index(rng, strict_index=True)


@pytest.mark.parametrize('use_slice_index', [True, False])
@pytest.mark.parametrize('add_offset', [True, False])
def test_imagedata_slice_index_integer(
    appended_images, appended_images_with_offset, add_offset, use_slice_index
):
    meshes = appended_images_with_offset if add_offset else appended_images
    slice0, slice1, appended = meshes

    # Slice with integer
    z = 0
    sliced = appended.slice_index(k=z) if use_slice_index else appended[:, :, z]
    assert sliced == slice0

    # Slice with negative integer
    z = -1
    sliced = appended.slice_index(k=z) if use_slice_index else appended[:, :, z]
    assert sliced == slice1


@pytest.mark.parametrize('use_slice_index', [True, False])
@pytest.mark.parametrize('add_offset', [True, False])
def test_imagedata_slice_index_range(
    appended_images, appended_images_with_offset, add_offset, use_slice_index
):
    meshes = appended_images_with_offset if add_offset else appended_images
    slice0, slice1, appended = meshes
    x_dim, y_dim, z_dim = appended.dimensions

    # Slice with index range equal to dimensions
    lower = 0
    sliced = (
        appended.slice_index(i=[lower, x_dim], j=[lower, y_dim], k=[lower, z_dim])
        if use_slice_index
        else appended[lower:x_dim, lower:y_dim, lower:z_dim]
    )
    assert sliced == appended

    # Slice with unspecified start and stop index
    lower = 0
    upper_x = x_dim
    upper_z = z_dim
    sliced = (
        appended.slice_index(i=[None, upper_x], j=[lower, None], k=[lower, upper_z])
        if use_slice_index
        else appended[:upper_x, lower:, lower:upper_z]
    )
    assert sliced == appended


@pytest.mark.parametrize('use_slice_index', [True, False])
@pytest.mark.parametrize('add_offset', [True, False])
def test_imagedata_slice_index_range_upper_bounds(
    appended_images, appended_images_with_offset, add_offset, use_slice_index
):
    meshes = appended_images_with_offset if add_offset else appended_images
    slice0, slice1, appended = meshes
    x_dim, y_dim, z_dim = appended.dimensions

    # Slice with upper range larger than dimensions
    lower = 0
    extra = 2
    sliced = (
        appended.slice_index(
            i=[lower, x_dim + extra], j=[lower, y_dim + extra], k=[lower, z_dim + extra]
        )
        if use_slice_index
        else appended[lower : x_dim + extra, lower : y_dim + extra, lower : z_dim + extra]
    )
    assert sliced == appended


@pytest.mark.parametrize('use_slice_index', [True, False])
@pytest.mark.parametrize('add_offset', [True, False])
def test_imagedata_slice_index_negative_range(
    appended_images, appended_images_with_offset, add_offset, use_slice_index
):
    meshes = appended_images_with_offset if add_offset else appended_images
    slice0, slice1, appended = meshes
    x_dim, y_dim, z_dim = appended.dimensions

    # Slice with negative stop index
    lower = 0
    upper = -1
    sliced_stop = (
        appended.slice_index(i=[lower, upper], j=[lower, upper], k=[lower, upper])
        if use_slice_index
        else appended[lower:upper, lower:upper, lower:upper]
    )
    assert sliced_stop.dimensions == (x_dim + upper, y_dim + upper, z_dim + upper)

    # Slice with negative start index
    lower = -3
    upper = -1
    sliced_start = (
        appended.slice_index(i=[lower, upper], j=[lower, upper], k=[lower, upper])
        if use_slice_index
        else appended[lower:upper, lower:upper, lower:upper]
    )
    assert sliced_start.dimensions == (x_dim + upper, y_dim + upper, z_dim + upper)
    assert sliced_stop == sliced_start


@pytest.mark.parametrize('use_slice_index', [True, False])
@pytest.mark.parametrize('add_offset', [True, False])
def test_imagedata_slice_index_all_none(
    appended_images, appended_images_with_offset, add_offset, use_slice_index
):
    meshes = appended_images_with_offset if add_offset else appended_images
    _, _, appended = meshes

    if use_slice_index:
        match = 'No indices were provided for slicing.'
        with pytest.raises(TypeError, match=match):
            appended.slice_index()
    else:
        sliced = appended[:, :, :]
        assert sliced == appended


def test_slice_index_indexing_range():
    mesh = pv.ImageData(dimensions=(10, 11, 12))
    mesh['data'] = range(mesh.n_points)
    index = np.array((5, 6, 7))
    offset = (1, 2, 3)
    mesh.offset = offset

    sliced_dimensions = mesh.slice_index(*index, index_mode='dimensions')
    sliced_extent = mesh.slice_index(*(index + offset), index_mode='extent')
    assert sliced_dimensions == sliced_extent


def test_imagedata_getitem_raises(uniform):
    match = 'Exactly 3 slices must be specified, one for each IJK-coordinate axis.'
    with pytest.raises(IndexError, match=re.escape(match)):
        uniform[0]

    with pytest.raises(IndexError, match=re.escape(match)):
        uniform[:]

    match = (
        "index must be an instance of any type (<class 'int'>, <class 'tuple'>, "
        "<class 'list'>, <class 'slice'>). Got <class 'dict'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        uniform[{}, str, set()]

    match = 'Only contiguous slices with step=1 are supported.'
    with pytest.raises(ValueError, match=re.escape(match)):
        uniform[2::2, 0, 0]

    match = (
        'index 10 is out of bounds for axis 0 with size 10.\n'
        'Valid range of valid index values (inclusive) is [-10, 9].'
    )
    with pytest.raises(IndexError, match=re.escape(match)):
        uniform[uniform.dimensions[0], 0, 0]

    uniform.offset = [1, 1, 1]
    _ = uniform.slice_index(uniform.dimensions[0], index_mode='extent')
    match = (
        'index 11 is out of bounds for axis 0 with size 10.\n'
        'Valid range of valid index values (inclusive) is [-9, 10].'
    )
    with pytest.raises(IndexError, match=re.escape(match)):
        uniform.slice_index(uniform.dimensions[0] + uniform.offset[0], 0, 0, index_mode='extent')
