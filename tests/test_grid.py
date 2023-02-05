import os
import pathlib
import weakref

import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples
from pyvista._vtk import VTK9
from pyvista.core.errors import VTKVersionError
from pyvista.errors import AmbiguousDataError, MissingDataError
from pyvista.plotting import system_supports_plotting
from pyvista.utilities.misc import PyVistaDeprecationWarning

test_path = os.path.dirname(os.path.abspath(__file__))

# must be manually set until pytest adds parametrize with fixture feature
HEXBEAM_CELLS_BOOL = np.ones(40, dtype=bool)  # matches hexbeam.n_cells == 40
STRUCTGRID_CELLS_BOOL = np.ones(729, dtype=bool)  # struct_grid.n_cells == 729
STRUCTGRID_POINTS_BOOL = np.ones(1000, dtype=bool)  # struct_grid.n_points == 1000

pointsetmark = pytest.mark.skipif(
    pyvista.vtk_version_info < (9, 1, 0), reason="Requires VTK>=9.1.0 for a concrete PointSet class"
)


def test_volume(hexbeam):
    assert hexbeam.volume > 0.0


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_struct_example():
    # create and plot structured grid
    grid = examples.load_structured()
    grid.plot(off_screen=True)  # basic plot
    grid.plot_curvature(off_screen=True)


def test_init_from_polydata(sphere):
    unstruct_grid = pyvista.UnstructuredGrid(sphere)
    assert unstruct_grid.n_points == sphere.n_points
    assert unstruct_grid.n_cells == sphere.n_cells
    assert np.all(unstruct_grid.celltypes == 5)


def test_init_from_structured(struct_grid):
    unstruct_grid = pyvista.UnstructuredGrid(struct_grid)
    assert unstruct_grid.points.shape[0] == struct_grid.x.size
    assert np.all(unstruct_grid.celltypes == 12)


def test_init_from_unstructured(hexbeam):
    grid = pyvista.UnstructuredGrid(hexbeam, deep=True)
    grid.points += 1
    assert not np.any(grid.points == hexbeam.points)

    grid = pyvista.UnstructuredGrid(hexbeam)
    grid.points += 1
    assert np.array_equal(grid.points, hexbeam.points)


def test_init_from_numpy_arrays():
    offset = np.array([0, 9])
    cells = [[8, 0, 1, 2, 3, 4, 5, 6, 7], [8, 8, 9, 10, 11, 12, 13, 14, 15]]
    cells = np.array(cells).ravel()
    cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON])
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
    if VTK9:
        grid = pyvista.UnstructuredGrid(cells, cell_type, points)
    else:
        grid = pyvista.UnstructuredGrid(offset, cells, cell_type, points)

    assert grid.number_of_points == 16
    assert grid.number_of_cells == 2


def test_init_bad_input():
    with pytest.raises(TypeError):
        pyvista.UnstructuredGrid(np.array(1))

    with pytest.raises(TypeError):
        pyvista.UnstructuredGrid(np.array(1), np.array(1), np.array(1), 'woa')


def create_hex_example():
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
    cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON], np.int32)

    cell1 = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float32,
    )

    cell2 = np.array(
        [[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2], [0, 0, 3], [1, 0, 3], [1, 1, 3], [0, 1, 3]],
        dtype=np.float32,
    )

    points = np.vstack((cell1, cell2))
    offset = np.array([0, 9], np.int8)

    return offset, cells, cell_type, points


# Try both with and without an offset array
@pytest.mark.parametrize('specify_offset', [False, True])
def test_init_from_arrays(specify_offset):
    offset, cells, cell_type, points = create_hex_example()

    if VTK9:
        grid = pyvista.UnstructuredGrid(cells, cell_type, points, deep=False)
    else:
        if specify_offset:
            grid = pyvista.UnstructuredGrid(offset, cells, cell_type, points, deep=False)
        else:
            grid = pyvista.UnstructuredGrid(cells, cell_type, points, deep=False)

        assert np.allclose(grid.offset, offset)

    assert grid.n_cells == 2
    assert np.allclose(cells, grid.cells)

    if VTK9:
        assert np.allclose(grid.cell_connectivity, np.arange(16))
    else:
        with pytest.raises(VTKVersionError):
            grid.cell_connectivity


@pytest.mark.parametrize('multiple_cell_types', [False, True])
@pytest.mark.parametrize('flat_cells', [False, True])
def test_init_from_dict(multiple_cell_types, flat_cells):
    # Try mixed construction
    vtk8_offsets, vtk_cell_format, cell_type, points = create_hex_example()

    vtk9_offsets = np.array([0, 8, 16])
    cells_hex = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
    input_cells_dict = {vtk.VTK_HEXAHEDRON: cells_hex}

    if multiple_cell_types:
        cells_quad = np.array([[16, 17, 18, 19]])

        cell3 = np.array([[0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1]])

        points = np.vstack((points, cell3))
        input_cells_dict[vtk.VTK_QUAD] = cells_quad

        # Update expected vtk cell arrays
        vtk_cell_format = np.concatenate([vtk_cell_format, [4], np.squeeze(cells_quad)])
        vtk8_offsets = np.concatenate([vtk8_offsets, [18]])
        vtk9_offsets = np.concatenate([vtk9_offsets, [20]])
        cell_type = np.concatenate([cell_type, [vtk.VTK_QUAD]])

    if flat_cells:
        input_cells_dict = {k: v.reshape([-1]) for k, v in input_cells_dict.items()}

    grid = pyvista.UnstructuredGrid(input_cells_dict, points, deep=False)

    if VTK9:
        assert np.all(grid.offset == vtk9_offsets)
    else:
        assert np.all(grid.offset == vtk8_offsets)

    assert grid.n_cells == (3 if multiple_cell_types else 2)
    assert np.all(grid.cells == vtk_cell_format)

    if VTK9:
        assert np.allclose(
            grid.cell_connectivity, (np.arange(20) if multiple_cell_types else np.arange(16))
        )
    else:
        with pytest.raises(VTKVersionError):
            grid.cell_connectivity

    # Now fetch the arrays
    output_cells_dict = grid.cells_dict

    assert np.all(
        output_cells_dict[vtk.VTK_HEXAHEDRON].reshape([-1])
        == input_cells_dict[vtk.VTK_HEXAHEDRON].reshape([-1])
    )

    if multiple_cell_types:
        assert np.all(
            output_cells_dict[vtk.VTK_QUAD].reshape([-1])
            == input_cells_dict[vtk.VTK_QUAD].reshape([-1])
        )

    # Test for some errors
    # Invalid index (<0)
    input_cells_dict[vtk.VTK_HEXAHEDRON] -= 1

    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid(input_cells_dict, points, deep=False)

    # Restore
    input_cells_dict[vtk.VTK_HEXAHEDRON] += 1

    # Invalid index (>= nr_points)
    input_cells_dict[vtk.VTK_HEXAHEDRON].flat[0] = points.shape[0]

    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid(input_cells_dict, points, deep=False)

    input_cells_dict[vtk.VTK_HEXAHEDRON] -= 1

    # Incorrect size
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid(
            {vtk.VTK_HEXAHEDRON: cells_hex.reshape([-1])[:-1]}, points, deep=False
        )

    # Unknown cell type
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid({255: cells_hex}, points, deep=False)

    # Dynamic sizes cell type
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid({vtk.VTK_POLYGON: cells_hex.reshape([-1])}, points, deep=False)

    # Non-integer arrays
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid(
            {vtk.VTK_HEXAHEDRON: cells_hex.reshape([-1])[:-1].astype(np.float32)}, points
        )

    # Invalid point dimensions
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid(input_cells_dict, points[..., :-1])


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

    polyhedron_connectivity = [3, 5, 17, 18, 19, 20, 21, 4, 17, 18, 23, 22, 4, 17, 21, 26, 22]
    cells = np.array([len(polyhedron_connectivity)] + polyhedron_connectivity)
    cell_type = np.array([pyvista.CellType.POLYHEDRON])
    grid = pyvista.UnstructuredGrid(cells, cell_type, nodes)

    assert grid.n_cells == len(cell_type)
    assert grid.get_cell(0).type == pyvista.CellType.POLYHEDRON


def test_cells_dict_hexbeam_file():
    grid = pyvista.UnstructuredGrid(examples.hexbeamfile)
    cells = np.delete(grid.cells, np.arange(0, grid.cells.size, 9)).reshape([-1, 8])

    assert np.all(grid.cells_dict[vtk.VTK_HEXAHEDRON] == cells)


def test_cells_dict_variable_length():
    cells_poly = np.concatenate([[5], np.arange(5)])
    cells_types = np.array([vtk.VTK_POLYGON])
    points = np.random.normal(size=(5, 3))
    grid = pyvista.UnstructuredGrid(cells_poly, cells_types, points)

    # Dynamic sizes cell types are currently unsupported
    with pytest.raises(ValueError):
        grid.cells_dict

    grid.celltypes[:] = 255
    # Unknown cell types
    with pytest.raises(ValueError):
        grid.cells_dict


def test_cells_dict_empty_grid():
    grid = pyvista.UnstructuredGrid()
    assert grid.cells_dict is None


def test_cells_dict_alternating_cells():
    cells = np.concatenate([[4], [1, 2, 3, 4], [3], [0, 1, 2], [4], [0, 1, 5, 6]])
    cells_types = np.array([vtk.VTK_QUAD, vtk.VTK_TRIANGLE, vtk.VTK_QUAD])
    points = np.random.normal(size=(3 + 2 * 2, 3))
    grid = pyvista.UnstructuredGrid(cells, cells_types, points)

    cells_dict = grid.cells_dict

    if VTK9:
        assert np.all(grid.offset == np.array([0, 4, 7, 11]))
    else:
        assert np.all(grid.offset == np.array([0, 5, 9]))

    assert np.all(cells_dict[vtk.VTK_QUAD] == np.array([cells[1:5], cells[-4:]]))
    assert np.all(cells_dict[vtk.VTK_TRIANGLE] == [0, 1, 2])


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
    assert (hexbeam.celltypes == vtk.VTK_TETRA).all()


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pyvista.pointset.UnstructuredGrid._WRITERS)
def test_save(extension, binary, tmpdir, hexbeam):
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp.{extension}'))
    hexbeam.save(filename, binary)

    grid = pyvista.UnstructuredGrid(filename)
    assert grid.cells.shape == hexbeam.cells.shape
    assert grid.points.shape == hexbeam.points.shape

    grid = pyvista.read(filename)
    assert grid.cells.shape == hexbeam.cells.shape
    assert grid.points.shape == hexbeam.points.shape
    assert isinstance(grid, pyvista.UnstructuredGrid)


def test_pathlib_read_write(tmpdir, hexbeam):
    path = pathlib.Path(str(tmpdir.mkdir("tmpdir").join('tmp.vtk')))
    assert not path.is_file()
    hexbeam.save(path)
    assert path.is_file()

    grid = pyvista.UnstructuredGrid(path)
    assert grid.cells.shape == hexbeam.cells.shape
    assert grid.points.shape == hexbeam.points.shape

    grid = pyvista.read(path)
    assert grid.cells.shape == hexbeam.cells.shape
    assert grid.points.shape == hexbeam.points.shape
    assert isinstance(grid, pyvista.UnstructuredGrid)


def test_init_bad_filename():
    filename = os.path.join(test_path, 'test_grid.py')
    with pytest.raises(IOError):
        pyvista.UnstructuredGrid(filename)

    with pytest.raises(FileNotFoundError):
        pyvista.UnstructuredGrid('not a file')


def test_save_bad_extension():
    with pytest.raises(FileNotFoundError):
        pyvista.UnstructuredGrid('file.abc')


def test_linear_copy(hexbeam):
    # need a grid with quadratic cells
    lgrid = hexbeam.linear_copy()
    assert np.all(lgrid.celltypes < 20)


def test_linear_copy_surf_elem():
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 6, 8, 9, 10, 11, 12, 13], np.int32)
    celltypes = np.array([vtk.VTK_QUADRATIC_QUAD, vtk.VTK_QUADRATIC_TRIANGLE], np.uint8)

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
    if VTK9:
        grid = pyvista.UnstructuredGrid(cells, celltypes, points, deep=False)
    else:
        offset = np.array([0, 9])
        grid = pyvista.UnstructuredGrid(offset, cells, celltypes, points, deep=False)

    lgrid = grid.linear_copy()

    qfilter = vtk.vtkMeshQuality()
    qfilter.SetInputData(lgrid)
    qfilter.Update()
    qual = pyvista.wrap(qfilter.GetOutput())['Quality']
    assert np.allclose(qual, [1, 1.4], atol=0.01)


def test_extract_cells(hexbeam):
    ind = [1, 2, 3]
    part_beam = hexbeam.extract_cells(ind)
    assert part_beam.n_cells == len(ind)
    assert part_beam.n_points < hexbeam.n_points
    assert np.allclose(part_beam.cell_data['vtkOriginalCellIds'], ind)

    mask = np.zeros(hexbeam.n_cells, dtype=bool)
    mask[ind] = True
    part_beam = hexbeam.extract_cells(mask)
    assert part_beam.n_cells == len(ind)
    assert part_beam.n_points < hexbeam.n_points
    assert np.allclose(part_beam.cell_data['vtkOriginalCellIds'], ind)

    ind = np.vstack(([1, 2], [4, 5]))[:, 0]
    part_beam = hexbeam.extract_cells(ind)


def test_merge(hexbeam):
    grid = hexbeam.copy()
    grid.points[:, 0] += 1
    unmerged = grid.merge(hexbeam, inplace=False, merge_points=False)

    grid.merge(hexbeam, inplace=True, merge_points=True)
    assert grid.n_points > hexbeam.n_points
    assert grid.n_points < unmerged.n_points


def test_merge_not_main(hexbeam):
    grid = hexbeam.copy()
    grid.points[:, 0] += 1
    unmerged = grid.merge(hexbeam, inplace=False, merge_points=False, main_has_priority=False)

    grid.merge(hexbeam, inplace=True, merge_points=True)
    assert grid.n_points > hexbeam.n_points
    assert grid.n_points < unmerged.n_points


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
    with pytest.raises(TypeError, match="Invalid parameters"):
        pyvista.StructuredGrid(['a', 'b', 'c'])
    with pytest.raises(ValueError, match="Too many args"):
        pyvista.StructuredGrid([0, 1], [0, 1], [0, 1], [0, 1])


def test_init_structured(struct_grid):
    xrng = np.arange(-10, 10, 2, dtype=np.float32)
    yrng = np.arange(-10, 10, 2, dtype=np.float32)
    zrng = np.arange(-10, 10, 2, dtype=np.float32)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    grid = pyvista.StructuredGrid(x, y, z)
    assert np.allclose(struct_grid.x, x)
    assert np.allclose(struct_grid.y, y)
    assert np.allclose(struct_grid.z, z)

    grid_a = pyvista.StructuredGrid(grid, deep=True)
    grid_a.points += 1
    assert not np.any(grid_a.points == grid.points)

    grid_a = pyvista.StructuredGrid(grid)
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
    source = np.random.rand(100, 3)
    mesh = pyvista.PolyData(source)
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_polydata_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.PolyData()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_structured_mesh_init(structured_points):
    source, dims = structured_points
    mesh = pyvista.StructuredGrid(source)
    mesh.dimensions = dims
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_structured_mesh_points_setter(structured_points):
    source, dims = structured_points
    mesh = pyvista.StructuredGrid()
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
    source = np.random.rand(100, 3)
    mesh = pyvista.PointSet(source)
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


@pointsetmark
def test_no_copy_pointset_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.PointSet()
    mesh.points = source
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)


def test_no_copy_unstructured_grid_points_setter():
    source = np.random.rand(100, 3)
    mesh = pyvista.UnstructuredGrid()
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
    mesh = pyvista.RectilinearGrid(xrng, yrng, zrng)
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

    with pytest.raises(RuntimeError):
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
    with pytest.raises(ValueError):
        pyvista.StructuredGrid(x, y, z)


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', pyvista.pointset.StructuredGrid._WRITERS)
def test_save_structured(extension, binary, tmpdir, struct_grid):
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp.{extension}'))
    struct_grid.save(filename, binary)

    grid = pyvista.StructuredGrid(filename)
    assert grid.x.shape == struct_grid.y.shape
    assert grid.n_cells
    assert grid.points.shape == struct_grid.points.shape

    grid = pyvista.read(filename)
    assert grid.x.shape == struct_grid.y.shape
    assert grid.n_cells
    assert grid.points.shape == struct_grid.points.shape
    assert isinstance(grid, pyvista.StructuredGrid)


def test_load_structured_bad_filename():
    with pytest.raises(FileNotFoundError):
        pyvista.StructuredGrid('not a file')

    filename = os.path.join(test_path, 'test_grid.py')
    with pytest.raises(IOError):
        pyvista.StructuredGrid(filename)


def test_instantiate_by_filename():
    ex = examples

    # actual mapping of example file to datatype
    fname_to_right_type = {
        ex.antfile: pyvista.PolyData,
        ex.planefile: pyvista.PolyData,
        ex.hexbeamfile: pyvista.UnstructuredGrid,
        ex.spherefile: pyvista.PolyData,
        ex.uniformfile: pyvista.UniformGrid,
        ex.rectfile: pyvista.RectilinearGrid,
    }

    # a few combinations of wrong type
    fname_to_wrong_type = {
        ex.antfile: pyvista.UnstructuredGrid,  # actual data is PolyData
        ex.planefile: pyvista.StructuredGrid,  # actual data is PolyData
        ex.rectfile: pyvista.UnstructuredGrid,  # actual data is StructuredGrid
    }

    # load the files into the right types
    for fname, right_type in fname_to_right_type.items():
        data = right_type(fname)
        assert data.n_points > 0

    # load the files into the wrong types
    for fname, wrong_type in fname_to_wrong_type.items():
        with pytest.raises(ValueError):
            data = wrong_type(fname)


def test_create_rectilinear_grid_from_specs():
    # 3D example
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 5)
    zrng = np.arange(-10, 10, 1)
    grid = pyvista.RectilinearGrid(xrng)
    assert grid.n_cells == 9
    assert grid.n_points == 10
    grid = pyvista.RectilinearGrid(xrng, yrng)
    assert grid.n_cells == 9 * 3
    assert grid.n_points == 10 * 4
    grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
    assert grid.n_cells == 9 * 3 * 19
    assert grid.n_points == 10 * 4 * 20
    assert grid.bounds == (-10.0, 8.0, -10.0, 5.0, -10.0, 9.0)
    # 2D example
    cell_spacings = np.array([1.0, 1.0, 2.0, 2.0, 5.0, 10.0])
    x_coordinates = np.cumsum(cell_spacings)
    y_coordinates = np.cumsum(cell_spacings)
    grid = pyvista.RectilinearGrid(x_coordinates, y_coordinates)
    assert grid.n_cells == 5 * 5
    assert grid.n_points == 6 * 6
    assert grid.bounds == (1.0, 21.0, 1.0, 21.0, 0.0, 0.0)


def test_create_rectilinear_after_init():
    x = np.array([0, 1, 2])
    y = np.array([0, 5, 8])
    z = np.array([3, 2, 1])
    grid = pyvista.RectilinearGrid()
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
    grid = pyvista.read(examples.rectfile)
    assert grid.n_cells == 16146
    assert grid.n_points == 18144
    assert grid.bounds == (-350.0, 1350.0, -400.0, 1350.0, -850.0, 0.0)
    assert grid.n_arrays == 1


def test_read_rectilinear_grid_from_pathlib():
    grid = pyvista.RectilinearGrid(pathlib.Path(examples.rectfile))
    assert grid.n_cells == 16146
    assert grid.n_points == 18144
    assert grid.bounds == (-350.0, 1350.0, -400.0, 1350.0, -850.0, 0.0)
    assert grid.n_arrays == 1


def test_raise_rectilinear_grid_non_unique():
    rng_uniq = np.arange(4.0)
    rng_dupe = np.array([0, 1, 2, 2], dtype=float)
    with pytest.raises(ValueError, match="Array contains duplicate values"):
        pyvista.RectilinearGrid(rng_dupe, check_duplicates=True)
    with pytest.raises(ValueError, match="Array contains duplicate values"):
        pyvista.RectilinearGrid(rng_uniq, rng_dupe, check_duplicates=True)
    with pytest.raises(ValueError, match="Array contains duplicate values"):
        pyvista.RectilinearGrid(rng_uniq, rng_uniq, rng_dupe, check_duplicates=True)


def test_cast_rectilinear_grid():
    grid = pyvista.read(examples.rectfile)
    structured = grid.cast_to_structured_grid()
    assert isinstance(structured, pyvista.StructuredGrid)
    assert structured.n_points == grid.n_points
    assert structured.n_cells == grid.n_cells
    assert np.allclose(structured.points, grid.points)
    for k, v in grid.point_data.items():
        assert np.allclose(structured.point_data[k], v)
    for k, v in grid.cell_data.items():
        assert np.allclose(structured.cell_data[k], v)


def test_create_uniform_grid_from_specs():
    # empty
    grid = pyvista.UniformGrid()

    # create UniformGrid
    dims = (10, 10, 10)
    grid = pyvista.UniformGrid(dimensions=dims)  # Using default spacing and origin
    assert grid.dimensions == dims
    assert grid.extent == (0, 9, 0, 9, 0, 9)
    assert grid.origin == (0.0, 0.0, 0.0)
    assert grid.spacing == (1.0, 1.0, 1.0)

    # Using default origin
    spacing = (2, 1, 5)
    grid = pyvista.UniformGrid(dimensions=dims, spacing=spacing)
    assert grid.dimensions == dims
    assert grid.origin == (0.0, 0.0, 0.0)
    assert grid.spacing == spacing
    origin = (10, 35, 50)

    # Everything is specified
    grid = pyvista.UniformGrid(dimensions=dims, spacing=spacing, origin=origin)
    assert grid.dimensions == dims
    assert grid.origin == origin
    assert grid.spacing == spacing

    # ensure negative spacing is not allowed
    with pytest.raises(ValueError, match="Spacing must be non-negative"):
        grid = pyvista.UniformGrid(dimensions=dims, spacing=(-1, 1, 1))

    # all args (deprecated)
    with pytest.warns(
        PyVistaDeprecationWarning, match="Behavior of pyvista.UniformGrid has changed"
    ):
        grid = pyvista.UniformGrid(dims, origin, spacing)
        assert grid.dimensions == dims
        assert grid.origin == origin
        assert grid.spacing == spacing

    # just dims (deprecated)
    with pytest.warns(
        PyVistaDeprecationWarning, match="Behavior of pyvista.UniformGrid has changed"
    ):
        grid = pyvista.UniformGrid(dims)
        assert grid.dimensions == dims

    with pytest.warns(
        PyVistaDeprecationWarning,
        match='`dims` argument is deprecated. Please use `dimensions`.',
    ):
        grid = pyvista.UniformGrid(dims=dims)
    with pytest.raises(TypeError):
        grid = pyvista.UniformGrid(dimensions=dims, dims=dims)

    # uniform grid from a uniform grid
    grid = pyvista.UniformGrid(dimensions=dims, spacing=spacing, origin=origin)
    grid_from_grid = pyvista.UniformGrid(grid)
    assert grid == grid_from_grid

    # and is a copy
    grid.origin = (0, 0, 0)
    assert grid != grid_from_grid


def test_uniform_grid_invald_args():
    with pytest.warns(
        PyVistaDeprecationWarning, match="Behavior of pyvista.UniformGrid has changed"
    ):
        pyvista.UniformGrid((1, 1, 1))

    with pytest.raises(TypeError):
        pyvista.UniformGrid(1)


def test_uniform_setters():
    grid = pyvista.UniformGrid()
    grid.dimensions = (10, 10, 10)
    assert grid.GetDimensions() == (10, 10, 10)
    assert grid.dimensions == (10, 10, 10)
    grid.spacing = (5, 2, 1)
    assert grid.GetSpacing() == (5, 2, 1)
    assert grid.spacing == (5, 2, 1)
    grid.origin = (6, 27.7, 19.8)
    assert grid.GetOrigin() == (6, 27.7, 19.8)
    assert grid.origin == (6, 27.7, 19.8)


def test_create_uniform_grid_from_file():
    grid = examples.load_uniform()
    assert grid.n_cells == 729
    assert grid.n_points == 1000
    assert grid.bounds == (0.0, 9.0, 0.0, 9.0, 0.0, 9.0)
    assert grid.n_arrays == 2
    assert grid.dimensions == (10, 10, 10)


def test_read_uniform_grid_from_file():
    grid = pyvista.read(examples.uniformfile)
    assert grid.n_cells == 729
    assert grid.n_points == 1000
    assert grid.bounds == (0.0, 9.0, 0.0, 9.0, 0.0, 9.0)
    assert grid.n_arrays == 2
    assert grid.dimensions == (10, 10, 10)


def test_read_uniform_grid_from_pathlib():
    grid = pyvista.UniformGrid(pathlib.Path(examples.uniformfile))
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
    rectilinear = grid.cast_to_rectilinear_grid()
    assert rectilinear.n_points == grid.n_points
    assert rectilinear.n_arrays == grid.n_arrays
    assert rectilinear.bounds == grid.bounds


def test_uniform_grid_to_tetrahedra():
    grid = pyvista.UniformGrid(dimensions=(2, 2, 2))
    ugrid = grid.to_tetrahedra()
    assert ugrid.n_cells == 5


def test_fft_and_rfft(noise_2d):
    grid = pyvista.UniformGrid(dimensions=(10, 10, 1))
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
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp.{extension}'))
    ogrid = examples.load_rectilinear()
    ogrid.save(filename, binary)
    grid = pyvista.RectilinearGrid(filename)
    assert grid.n_cells == ogrid.n_cells
    assert np.allclose(grid.x, ogrid.x)
    assert np.allclose(grid.y, ogrid.y)
    assert np.allclose(grid.z, ogrid.z)
    assert grid.dimensions == ogrid.dimensions
    grid = pyvista.read(filename)
    assert isinstance(grid, pyvista.RectilinearGrid)
    assert grid.n_cells == ogrid.n_cells
    assert np.allclose(grid.x, ogrid.x)
    assert np.allclose(grid.y, ogrid.y)
    assert np.allclose(grid.z, ogrid.z)
    assert grid.dimensions == ogrid.dimensions


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['.vtk', '.vti'])
def test_save_uniform(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp.{extension}'))
    ogrid = examples.load_uniform()
    ogrid.save(filename, binary)
    grid = pyvista.UniformGrid(filename)
    assert grid.n_cells == ogrid.n_cells
    assert grid.origin == ogrid.origin
    assert grid.spacing == ogrid.spacing
    assert grid.dimensions == ogrid.dimensions
    grid = pyvista.read(filename)
    assert isinstance(grid, pyvista.UniformGrid)
    assert grid.n_cells == ogrid.n_cells
    assert grid.origin == ogrid.origin
    assert grid.spacing == ogrid.spacing
    assert grid.dimensions == ogrid.dimensions


def test_grid_points():
    """Test the points methods on UniformGrid and RectilinearGrid"""
    # test creation of 2d grids
    x = y = range(3)
    z = [0]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]
    grid = pyvista.UniformGrid()
    with pytest.raises(AttributeError):
        grid.points = points
    grid.origin = (0.0, 0.0, 0.0)
    grid.dimensions = (3, 3, 1)
    grid.spacing = (1, 1, 1)
    assert grid.n_points == 9
    assert grid.n_cells == 4
    assert np.allclose(grid.points, points)

    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    )
    grid = pyvista.UniformGrid()
    grid.dimensions = [2, 2, 2]
    grid.spacing = [1, 1, 1]
    grid.origin = [0.0, 0.0, 0.0]
    assert np.allclose(np.unique(grid.points, axis=0), np.unique(points, axis=0))
    opts = np.c_[grid.x, grid.y, grid.z]
    assert np.allclose(np.unique(opts, axis=0), np.unique(points, axis=0))

    # Now test rectilinear grid
    grid = pyvista.RectilinearGrid()
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
        grid.points, np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]
    )


def test_grid_extract_selection_points(struct_grid):
    grid = pyvista.UnstructuredGrid(struct_grid)
    sub_grid = grid.extract_points([0])
    assert sub_grid.n_cells == 1

    sub_grid = grid.extract_points(range(100))
    assert sub_grid.n_cells > 1


def test_gaussian_smooth(hexbeam):
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
def test_remove_cells(ind, hexbeam):
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
    with pytest.raises(ValueError):
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
    uni_grid = pyvista.UniformGrid(dimensions=[10, 10, 10])
    with pytest.raises(ValueError):
        uni_grid.extent = [0, 1]

    extent = [0, 1, 0, 1, 0, 1]
    uni_grid.extent = extent
    assert np.array_equal(uni_grid.extent, extent)


@pytest.mark.needs_vtk9
def test_UnstructuredGrid_cast_to_explicit_structured_grid():
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


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_init():
    grid = examples.load_explicit_structured()
    assert isinstance(grid, pyvista.ExplicitStructuredGrid)
    assert grid.n_cells == 120
    assert grid.n_points == 210
    assert grid.bounds == (0.0, 80.0, 0.0, 50.0, 0.0, 6.0)
    assert repr(grid) == str(grid)
    assert 'N Cells' in str(grid)
    assert 'N Points' in str(grid)
    assert 'N Arrays' in str(grid)


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_cast_to_unstructured_grid():
    block_i = np.fromstring(
        '''
        0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
        1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1
        2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2
        3 0 1 2 3 0 1 2 3
        ''',
        sep=' ',
        dtype=int,
    )

    block_j = np.fromstring(
        '''
        0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4
        4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3
        3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2
        2 3 3 3 3 4 4 4 4
        ''',
        sep=' ',
        dtype=int,
    )

    block_k = np.fromstring(
        '''
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3
        3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5
        5 5 5 5 5 5 5 5 5
        ''',
        sep=' ',
        dtype=int,
    )

    grid = examples.load_explicit_structured()
    grid = grid.cast_to_unstructured_grid()
    assert isinstance(grid, pyvista.UnstructuredGrid)
    assert 'BLOCK_I' in grid.cell_data
    assert 'BLOCK_J' in grid.cell_data
    assert 'BLOCK_K' in grid.cell_data
    assert np.array_equal(grid.cell_data['BLOCK_I'], block_i)
    assert np.array_equal(grid.cell_data['BLOCK_J'], block_j)
    assert np.array_equal(grid.cell_data['BLOCK_K'], block_k)


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_save():
    grid = examples.load_explicit_structured()
    grid = grid.hide_cells(range(80, 120))
    grid.save('grid.vtu')
    grid = pyvista.ExplicitStructuredGrid('grid.vtu')
    assert grid.n_cells == 120
    assert grid.n_points == 210
    assert grid.bounds == (0.0, 80.0, 0.0, 50.0, 0.0, 6.0)
    assert np.count_nonzero(grid.cell_data['vtkGhostType']) == 40
    os.remove('grid.vtu')


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_hide_cells():
    ghost = np.asarray(
        '''
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32
    32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32
    '''.split(),
        dtype=np.uint8,
    )

    grid = examples.load_explicit_structured()

    copy = grid.hide_cells(range(80, 120))
    assert isinstance(copy, pyvista.ExplicitStructuredGrid)
    assert 'vtkGhostType' in copy.cell_data
    assert 'vtkGhostType' not in grid.cell_data
    assert np.array_equal(copy.cell_data['vtkGhostType'], ghost)

    out = grid.hide_cells(range(80, 120), inplace=True)
    assert out is grid
    assert 'vtkGhostType' in grid.cell_data
    assert np.array_equal(grid.cell_data['vtkGhostType'], ghost)


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_show_cells():
    grid = examples.load_explicit_structured()
    grid.hide_cells(range(80, 120), inplace=True)

    copy = grid.show_cells()
    assert isinstance(copy, pyvista.ExplicitStructuredGrid)
    assert 'vtkGhostType' in copy.cell_data
    assert np.count_nonzero(copy.cell_data['vtkGhostType']) == 0
    assert np.count_nonzero(grid.cell_data['vtkGhostType']) == 40

    out = grid.show_cells(inplace=True)
    assert out is grid
    assert np.count_nonzero(grid.cell_data['vtkGhostType']) == 0


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_dimensions():
    grid = examples.load_explicit_structured()
    assert isinstance(grid.dimensions, tuple)
    assert isinstance(grid.dimensions[0], int)
    assert len(grid.dimensions) == 3
    assert grid.dimensions == (5, 6, 7)


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_visible_bounds():
    grid = examples.load_explicit_structured()
    grid = grid.hide_cells(range(80, 120))
    assert isinstance(grid.visible_bounds, tuple)
    assert all(isinstance(x, float) for x in grid.visible_bounds)
    assert len(grid.visible_bounds) == 6
    assert grid.visible_bounds == (0.0, 80.0, 0.0, 50.0, 0.0, 4.0)


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_cell_id():
    grid = examples.load_explicit_structured()

    ind = grid.cell_id((3, 4, 0))
    assert np.issubdtype(ind, np.integer)
    assert ind == 19

    ind = grid.cell_id([(3, 4, 0), (3, 2, 1), (1, 0, 2), (2, 3, 2)])
    assert isinstance(ind, np.ndarray)
    assert np.issubdtype(ind.dtype, np.integer)
    assert np.array_equal(ind, [19, 31, 41, 54])


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_cell_coords():
    grid = examples.load_explicit_structured()

    coords = grid.cell_coords(19)
    assert isinstance(coords, tuple)
    assert all(np.issubdtype(c, np.integer) for c in coords)
    assert coords == (3, 4, 0)

    coords = grid.cell_coords((19, 31, 41, 54))
    assert isinstance(coords, np.ndarray)
    assert np.issubdtype(coords.dtype, np.integer)
    assert np.array_equal(coords, [(3, 4, 0), (3, 2, 1), (1, 0, 2), (2, 3, 2)])


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_neighbors():
    grid = examples.load_explicit_structured()

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


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_compute_connectivity():
    connectivity = np.asarray(
        '''
    42 43 43 41 46 47 47 45 46 47 47 45 46 47 47 45 38 39 39 37 58 59 59 57
    62 63 63 61 62 63 63 61 62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61
    62 63 63 61 62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61 62 63 63 61
    62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61 62 63 63 61 62 63 63 61
    54 55 55 53 26 27 27 25 30 31 31 29 30 31 31 29 30 31 31 29 22 23 23 21
    '''.split(),
        dtype=int,
    )

    grid = examples.load_explicit_structured()
    assert 'ConnectivityFlags' not in grid.cell_data

    copy = grid.compute_connectivity()
    assert isinstance(copy, pyvista.ExplicitStructuredGrid)
    assert 'ConnectivityFlags' in copy.cell_data
    assert 'ConnectivityFlags' not in grid.cell_data
    assert np.array_equal(copy.cell_data['ConnectivityFlags'], connectivity)

    out = grid.compute_connectivity(inplace=True)
    assert out is grid
    assert 'ConnectivityFlags' in grid.cell_data
    assert np.array_equal(grid.cell_data['ConnectivityFlags'], connectivity)


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_compute_connections():
    connections = np.asarray(
        '''
    3 4 4 3 4 5 5 4 4 5 5 4 4 5 5 4 3 4 4 3 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4
    5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4 5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6
    6 5 4 5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4 5 5 4 3 4 4 3 4 5 5 4 4 5 5
    4 4 5 5 4 3 4 4 3
    '''.split(),
        dtype=int,
    )

    grid = examples.load_explicit_structured()
    assert 'number_of_connections' not in grid.cell_data

    copy = grid.compute_connections()
    assert isinstance(copy, pyvista.ExplicitStructuredGrid)
    assert 'number_of_connections' in copy.cell_data
    assert 'number_of_connections' not in grid.cell_data
    assert np.array_equal(copy.cell_data['number_of_connections'], connections)

    grid.compute_connections(inplace=True)
    assert 'number_of_connections' in grid.cell_data
    assert np.array_equal(grid.cell_data['number_of_connections'], connections)


@pytest.mark.needs_vtk9
def test_ExplicitStructuredGrid_raise_init():
    with pytest.raises(ValueError, match="Too many args"):
        pyvista.ExplicitStructuredGrid(1, 2, True)


def test_copy_no_copy_wrap_object(datasets):
    for dataset in datasets:
        # different dataset types have different copy behavior for points
        # use point data which is common
        dataset["data"] = np.ones(dataset.n_points)
        new_dataset = type(dataset)(dataset)
        new_dataset["data"] += 1
        assert np.array_equal(new_dataset["data"], dataset["data"])

    for dataset in datasets:
        # different dataset types have different copy behavior for points
        # use point data which is common
        dataset["data"] = np.ones(dataset.n_points)
        new_dataset = type(dataset)(dataset, deep=True)
        new_dataset["data"] += 1
        assert not np.any(new_dataset["data"] == dataset["data"])


@pytest.mark.needs_vtk9
def test_copy_no_copy_wrap_object_vtk9(datasets_vtk9):
    for dataset in datasets_vtk9:
        # different dataset tyoes have different copy behavior for points
        # use point data which is common
        dataset["data"] = np.ones(dataset.n_points)
        new_dataset = type(dataset)(dataset)
        new_dataset["data"] += 1
        assert np.array_equal(new_dataset["data"], dataset["data"])

    for dataset in datasets_vtk9:
        # different dataset tyoes have different copy behavior for points
        # use point data which is common
        dataset["data"] = np.ones(dataset.n_points)
        new_dataset = type(dataset)(dataset, deep=True)
        new_dataset["data"] += 1
        assert not np.any(new_dataset["data"] == dataset["data"])
