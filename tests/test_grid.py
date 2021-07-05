import pathlib
import os
import weakref

import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples
from pyvista.plotting import system_supports_plotting

test_path = os.path.dirname(os.path.abspath(__file__))

VTK9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9

# must be manually set until pytest adds parametrize with fixture feature
HEXBEAM_CELLS_BOOL = np.ones(40, np.bool_)  # matches hexbeam.n_cells == 40
STRUCTGRID_CELLS_BOOL = np.ones(729, np.bool_)  # struct_grid.n_cells == 729


def test_volume(hexbeam):
    assert hexbeam.volume > 0.0


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_struct_example():
    # create and plot structured grid
    grid = examples.load_structured()
    grid.plot(off_screen=True)  # basic plot
    grid.plot_curvature(off_screen=True)


def test_init_from_structured(struct_grid):
    unstruct_grid = pyvista.UnstructuredGrid(struct_grid)
    assert unstruct_grid.points.shape[0] == struct_grid.x.size
    assert np.all(unstruct_grid.celltypes == 12)


def test_init_from_unstructured(hexbeam):
    grid = pyvista.UnstructuredGrid(hexbeam, deep=True)
    grid.points += 1
    assert not np.any(grid.points == hexbeam.points)


def test_init_from_numpy_arrays():
    offset = np.array([0, 9])
    cells = [
        [8, 0, 1, 2, 3, 4, 5, 6, 7],
        [8, 8, 9, 10, 11, 12, 13, 14, 15]
    ]
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
        ]
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
        ]
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
        unstruct_grid = pyvista.UnstructuredGrid(np.array(1))

    with pytest.raises(TypeError):
        unstruct_grid = pyvista.UnstructuredGrid(np.array(1),
                                                 np.array(1),
                                                 np.array(1),
                                                 'woa')

def create_hex_example():
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
    cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON], np.int32)

    cell1 = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [1, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [1, 1, 1],
                      [0, 1, 1]])

    cell2 = np.array([[0, 0, 2],
                      [1, 0, 2],
                      [1, 1, 2],
                      [0, 1, 2],
                      [0, 0, 3],
                      [1, 0, 3],
                      [1, 1, 3],
                      [0, 1, 3]])

    points = np.vstack((cell1, cell2)).astype(np.int32)
    offset = np.array([0, 9], np.int8)

    return offset, cells, cell_type, points

#Try both with and without an offset array
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
        with pytest.raises(AttributeError):
            grid.cell_connectivity

@pytest.mark.parametrize('multiple_cell_types', [False, True])
@pytest.mark.parametrize('flat_cells', [False, True])
def test_init_from_dict(multiple_cell_types, flat_cells):
    #Try mixed construction
    vtk8_offsets, vtk_cell_format, cell_type, points = create_hex_example()

    vtk9_offsets = np.array([0, 8, 16])
    cells_hex = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
    input_cells_dict = {vtk.VTK_HEXAHEDRON: cells_hex}
    
    if multiple_cell_types:
        cells_quad = np.array([[16, 17, 18, 19]])
    
        cell3 = np.array([[0, 0, -1],
                          [1, 0, -1],
                          [1, 1, -1],
                          [0, 1, -1]])


        points = np.vstack((points, cell3))
        input_cells_dict[vtk.VTK_QUAD] = cells_quad

        #Update expected vtk cell arrays
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
        assert np.allclose(grid.cell_connectivity, (np.arange(20) if multiple_cell_types else np.arange(16)))
    else:
        with pytest.raises(AttributeError):
            grid.cell_connectivity

    #Now fetch the arrays
    output_cells_dict = grid.cells_dict

    assert np.all(output_cells_dict[vtk.VTK_HEXAHEDRON].reshape([-1]) == input_cells_dict[vtk.VTK_HEXAHEDRON].reshape([-1]))

    if multiple_cell_types:
        assert np.all(output_cells_dict[vtk.VTK_QUAD].reshape([-1]) == input_cells_dict[vtk.VTK_QUAD].reshape([-1]))


    #Test for some errors
    #Invalid index (<0)
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
        pyvista.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells_hex.reshape([-1])[:-1]}, points, deep=False)

    # Unknown cell type
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid({255: cells_hex}, points, deep=False)

    # Dynamic sizes cell type
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid({vtk.VTK_POLYGON: cells_hex.reshape([-1])}, points, deep=False)

    # Non-integer arrays
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells_hex.reshape([-1])[:-1].astype(np.float32)}, points)

    # Invalid point dimensions
    with pytest.raises(ValueError):
        pyvista.UnstructuredGrid(input_cells_dict, points[..., :-1])


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
    points = np.random.normal(size=(3+2*2, 3))
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
    surf_ind = surf.point_arrays['vtkOriginalPointIds']
    assert np.allclose(surf_ind, hexbeam.surface_indices())


def test_extract_feature_edges(hexbeam):
    edges = hexbeam.extract_feature_edges(90)
    assert edges.n_points

    edges = hexbeam.extract_feature_edges(180)
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
        grid = pyvista.UnstructuredGrid(filename)

    with pytest.raises(FileNotFoundError):
        grid = pyvista.UnstructuredGrid('not a file')


def test_save_bad_extension():
    with pytest.raises(FileNotFoundError):
        grid = pyvista.UnstructuredGrid('file.abc')


def test_linear_copy(hexbeam):
    # need a grid with quadratic cells
    lgrid = hexbeam.linear_copy()
    assert np.all(lgrid.celltypes < 20)


def test_linear_copy_surf_elem():
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 6, 8, 9, 10, 11, 12, 13], np.int32)
    celltypes = np.array([vtk.VTK_QUADRATIC_QUAD, vtk.VTK_QUADRATIC_TRIANGLE],
                         np.uint8)

    cell0 = [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.5, 0.1, 0.0],
             [1.1, 0.5, 0.0],
             [0.5, 0.9, 0.0],
             [0.1, 0.5, 0.0]]

    cell1 = [[0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0],
             [0.5, 0.5, 1.0],
             [0.5, 0.0, 1.3],
             [0.7, 0.7, 1.3],
             [0.1, 0.1, 1.3]]

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
    assert np.allclose(part_beam.cell_arrays['vtkOriginalCellIds'], ind)

    mask = np.zeros(hexbeam.n_cells, np.bool_)
    mask[ind] = True
    part_beam = hexbeam.extract_cells(mask)
    assert part_beam.n_cells == len(ind)
    assert part_beam.n_points < hexbeam.n_points
    assert np.allclose(part_beam.cell_arrays['vtkOriginalCellIds'], ind)

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
    unmerged = grid.merge(hexbeam, inplace=False, merge_points=False,
                          main_has_priority=False)

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


def test_init_structured(struct_grid):
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 2)
    zrng = np.arange(-10, 10, 2)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    grid = pyvista.StructuredGrid(x, y, z)
    assert np.allclose(struct_grid.x, x)
    assert np.allclose(struct_grid.y, y)
    assert np.allclose(struct_grid.z, z)

    grid_a = pyvista.StructuredGrid(grid)
    assert np.allclose(grid_a.points, grid.points)


def test_slice_structured(struct_grid):
    sliced = struct_grid[1, :, 1:3]  # three different kinds of slices
    assert sliced.dimensions == [1, struct_grid.dimensions[1], 2]

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
        grid = pyvista.StructuredGrid(x, y, z)


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
        grid = pyvista.StructuredGrid(filename)


def test_instantiate_by_filename():
    ex = examples

    # actual mapping of example file to datatype
    fname_to_right_type = {
        ex.antfile: pyvista.PolyData,
        ex.planefile: pyvista.PolyData,
        ex.hexbeamfile: pyvista.UnstructuredGrid,
        ex.spherefile: pyvista.PolyData,
        ex.uniformfile: pyvista.UniformGrid,
        ex.rectfile: pyvista.RectilinearGrid
    }

    # a few combinations of wrong type
    fname_to_wrong_type = {
        ex.antfile: pyvista.UnstructuredGrid,   # actual data is PolyData
        ex.planefile: pyvista.StructuredGrid,   # actual data is PolyData
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
    assert grid.n_cells == 9*3
    assert grid.n_points == 10*4
    grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
    assert grid.n_cells == 9*3*19
    assert grid.n_points == 10*4*20
    assert grid.bounds == [-10.0,8.0, -10.0,5.0, -10.0,9.0]
    # 2D example
    cell_spacings = np.array([1., 1., 2., 2., 5., 10.])
    x_coordinates = np.cumsum(cell_spacings)
    y_coordinates = np.cumsum(cell_spacings)
    grid = pyvista.RectilinearGrid(x_coordinates, y_coordinates)
    assert grid.n_cells == 5*5
    assert grid.n_points == 6*6
    assert grid.bounds == [1.,21., 1.,21., 0.,0.]


def test_create_rectilinear_after_init():
    x = np.array([0,1,2])
    y = np.array([0,5,8])
    z = np.array([3,2,1])
    grid = pyvista.RectilinearGrid()
    grid.x = x
    assert grid.dimensions == [3, 1, 1]
    grid.y = y
    assert grid.dimensions == [3, 3, 1]
    grid.z = z
    assert grid.dimensions == [3, 3, 3]
    assert np.allclose(grid.x, x)
    assert np.allclose(grid.y, y)
    assert np.allclose(grid.z, z)


def test_create_rectilinear_grid_from_file():
    grid = examples.load_rectilinear()
    assert grid.n_cells == 16146
    assert grid.n_points == 18144
    assert grid.bounds == [-350.0,1350.0, -400.0,1350.0, -850.0,0.0]
    assert grid.n_arrays == 1


def test_read_rectilinear_grid_from_file():
    grid = pyvista.read(examples.rectfile)
    assert grid.n_cells == 16146
    assert grid.n_points == 18144
    assert grid.bounds == [-350.0,1350.0, -400.0,1350.0, -850.0,0.0]
    assert grid.n_arrays == 1


def test_read_rectilinear_grid_from_pathlib():
    grid = pyvista.RectilinearGrid(pathlib.Path(examples.rectfile))
    assert grid.n_cells == 16146
    assert grid.n_points == 18144
    assert grid.bounds == [-350.0, 1350.0, -400.0, 1350.0, -850.0, 0.0]
    assert grid.n_arrays == 1


def test_cast_rectilinear_grid():
    grid = pyvista.read(examples.rectfile)
    structured = grid.cast_to_structured_grid()
    assert isinstance(structured, pyvista.StructuredGrid)
    assert structured.n_points == grid.n_points
    assert structured.n_cells == grid.n_cells
    assert np.allclose(structured.points, grid.points)
    for k, v in grid.point_arrays.items():
        assert np.allclose(structured.point_arrays[k], v)
    for k, v in grid.cell_arrays.items():
        assert np.allclose(structured.cell_arrays[k], v)


def test_create_uniform_grid_from_specs():
    # create UniformGrid
    dims = [10, 10, 10]
    grid = pyvista.UniformGrid(dims) # Using default spacing and origin
    assert grid.dimensions == [10, 10, 10]
    assert grid.extent == [0, 9, 0, 9, 0, 9]
    assert grid.origin == [0.0, 0.0, 0.0]
    assert grid.spacing == [1.0, 1.0, 1.0]
    spacing = [2, 1, 5]
    grid = pyvista.UniformGrid(dims, spacing) # Using default origin
    assert grid.dimensions == [10, 10, 10]
    assert grid.origin == [0.0, 0.0, 0.0]
    assert grid.spacing == [2.0, 1.0, 5.0]
    origin = [10, 35, 50]
    grid = pyvista.UniformGrid(dims, spacing, origin) # Everything is specified
    assert grid.dimensions == [10, 10, 10]
    assert grid.origin == [10.0, 35.0, 50.0]
    assert grid.spacing == [2.0, 1.0, 5.0]
    assert grid.dimensions == [10, 10, 10]


def test_uniform_setters():
    grid = pyvista.UniformGrid()
    grid.dimensions = [10, 10, 10]
    assert grid.GetDimensions() == (10, 10, 10)
    assert grid.dimensions == [10, 10, 10]
    grid.spacing = [5, 2, 1]
    assert grid.GetSpacing() == (5, 2, 1)
    assert grid.spacing == [5, 2, 1]
    grid.origin = [6, 27.7, 19.8]
    assert grid.GetOrigin() == (6, 27.7, 19.8)
    assert grid.origin == [6, 27.7, 19.8]


def test_create_uniform_grid_from_file():
    grid = examples.load_uniform()
    assert grid.n_cells == 729
    assert grid.n_points == 1000
    assert grid.bounds == [0.0,9.0, 0.0,9.0, 0.0,9.0]
    assert grid.n_arrays == 2
    assert grid.dimensions == [10, 10, 10]


def test_read_uniform_grid_from_file():
    grid = pyvista.read(examples.uniformfile)
    assert grid.n_cells == 729
    assert grid.n_points == 1000
    assert grid.bounds == [0.0,9.0, 0.0,9.0, 0.0,9.0]
    assert grid.n_arrays == 2
    assert grid.dimensions == [10, 10, 10]


def test_read_uniform_grid_from_pathlib():
    grid = pyvista.UniformGrid(pathlib.Path(examples.uniformfile))
    assert grid.n_cells == 729
    assert grid.n_points == 1000
    assert grid.bounds == [0.0, 9.0, 0.0, 9.0, 0.0, 9.0]
    assert grid.n_arrays == 2
    assert grid.dimensions == [10, 10, 10]


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
    z = [0,]
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

    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1],
                       [1, 1, 1],
                       [0, 1, 1]])
    grid = pyvista.UniformGrid()
    grid.dimensions = [2, 2, 2]
    grid.spacing = [1, 1, 1]
    grid.origin = [0., 0., 0.]
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
    assert grid.dimensions == [3, 3, 2]
    assert np.allclose(grid.meshgrid, (xx, yy, zz))
    assert np.allclose(grid.points, np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')])


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


@pytest.mark.parametrize('ind', [range(10), np.arange(10),
                                 HEXBEAM_CELLS_BOOL])
def test_remove_cells(ind, hexbeam):
    grid_copy = hexbeam.copy()
    grid_copy.remove_cells(ind)
    assert grid_copy.n_cells < hexbeam.n_cells


@pytest.mark.parametrize('ind', [range(10), np.arange(10),
                                 HEXBEAM_CELLS_BOOL])
def test_remove_cells_not_inplace(ind, hexbeam):
    grid_copy = hexbeam.copy()  # copy to protect
    grid_w_removed = grid_copy.remove_cells(ind, inplace=False)
    assert grid_w_removed.n_cells < hexbeam.n_cells
    assert grid_copy.n_cells == hexbeam.n_cells


def test_remove_cells_invalid(hexbeam):
    grid_copy = hexbeam.copy()
    with pytest.raises(ValueError):
        grid_copy.remove_cells(np.ones(10, np.bool_))


@pytest.mark.parametrize('ind', [range(10), np.arange(10),
                                 STRUCTGRID_CELLS_BOOL])
def test_hide_cells(ind, struct_grid):
    sgrid_copy = struct_grid.copy()
    sgrid_copy.hide_cells(ind)
    assert sgrid_copy.HasAnyBlankCells()

    with pytest.raises(ValueError, match='Boolean array size must match'):
        sgrid_copy.hide_cells(np.ones(10, dtype=np.bool))



@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_UnstructuredGrid_cast_to_explicit_structured_grid():
    grid = examples.load_explicit_structured()
    grid.hide_cells(range(80, 120))
    grid = grid.cast_to_unstructured_grid()
    grid = grid.cast_to_explicit_structured_grid()
    assert grid.n_cells == 120
    assert grid.n_points == 210
    assert grid.bounds == [0.0, 80.0, 0.0, 50.0, 0.0, 6.0]
    assert 'BLOCK_I' in grid.cell_arrays
    assert 'BLOCK_J' in grid.cell_arrays
    assert 'BLOCK_K' in grid.cell_arrays
    assert 'vtkGhostType' in grid.cell_arrays
    assert np.count_nonzero(grid.cell_arrays['vtkGhostType']) == 40


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_init():
    grid = examples.load_explicit_structured()
    assert isinstance(grid, pyvista.ExplicitStructuredGrid)
    assert grid.n_cells == 120
    assert grid.n_points == 210
    assert grid.bounds == [0.0, 80.0, 0.0, 50.0, 0.0, 6.0]
    assert repr(grid) == str(grid)
    assert 'N Cells' in str(grid)
    assert 'N Points' in str(grid)
    assert 'N Arrays' in str(grid)


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_cast_to_unstructured_grid():
    block_i = np.asarray('''
    0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0
    1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1
    2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2
    3 0 1 2 3 0 1 2 3
    '''.split(), dtype=int)

    block_j = np.asarray('''
    0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4
    4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3
    3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 0 0 0 0 1 1 1 1 2 2 2
    2 3 3 3 3 4 4 4 4
    '''.split(), dtype=int)

    block_k = np.asarray('''
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3
    3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5
    5 5 5 5 5 5 5 5 5
    '''.split(), dtype=int)

    grid = examples.load_explicit_structured()
    grid = grid.cast_to_unstructured_grid()
    assert isinstance(grid, pyvista.UnstructuredGrid)
    assert 'BLOCK_I' in grid.cell_arrays
    assert 'BLOCK_J' in grid.cell_arrays
    assert 'BLOCK_K' in grid.cell_arrays
    assert np.array_equal(grid.cell_arrays['BLOCK_I'], block_i)
    assert np.array_equal(grid.cell_arrays['BLOCK_J'], block_j)
    assert np.array_equal(grid.cell_arrays['BLOCK_K'], block_k)


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_save():
    grid = examples.load_explicit_structured()
    grid.hide_cells(range(80, 120))
    grid.save('grid.vtu')
    grid = pyvista.ExplicitStructuredGrid('grid.vtu')
    assert grid.n_cells == 120
    assert grid.n_points == 210
    assert grid.bounds == [0.0, 80.0, 0.0, 50.0, 0.0, 6.0]
    assert np.count_nonzero(grid.cell_arrays['vtkGhostType']) == 40
    os.remove('grid.vtu')


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_hide_cells():
    ghost = np.asarray('''
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
     0  0  0  0  0  0  0  0 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32
    32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32
    '''.split(), dtype=np.uint8)

    grid = examples.load_explicit_structured()

    copy = grid.hide_cells(range(80, 120), inplace=False)
    assert isinstance(copy, pyvista.ExplicitStructuredGrid)
    assert 'vtkGhostType' in copy.cell_arrays
    assert 'vtkGhostType' not in grid.cell_arrays
    assert np.array_equal(copy.cell_arrays['vtkGhostType'], ghost)

    out = grid.hide_cells(range(80, 120), inplace=True)
    assert out is grid
    assert 'vtkGhostType' in grid.cell_arrays
    assert np.array_equal(grid.cell_arrays['vtkGhostType'], ghost)


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_show_cells():
    grid = examples.load_explicit_structured()
    grid.hide_cells(range(80, 120), inplace=True)

    copy = grid.show_cells(inplace=False)
    assert isinstance(copy, pyvista.ExplicitStructuredGrid)
    assert 'vtkGhostType' in copy.cell_arrays
    assert np.count_nonzero(copy.cell_arrays['vtkGhostType']) == 0
    assert np.count_nonzero(grid.cell_arrays['vtkGhostType']) == 40

    out = grid.show_cells(inplace=True)
    assert out is grid
    assert np.count_nonzero(grid.cell_arrays['vtkGhostType']) == 0


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_dimensions():
    grid = examples.load_explicit_structured()
    assert isinstance(grid.dimensions, np.ndarray)
    assert np.issubdtype(grid.dimensions.dtype, np.integer)
    assert grid.dimensions.shape == (3,)
    assert np.array_equal(grid.dimensions, [5, 6, 7])


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_visible_bounds():
    grid = examples.load_explicit_structured()
    grid.hide_cells(range(80, 120))
    assert isinstance(grid.visible_bounds, list)
    assert all(isinstance(x, float) for x in grid.visible_bounds)
    assert len(grid.visible_bounds) == 6
    assert grid.visible_bounds == [0.0, 80.0, 0.0, 50.0, 0.0, 4.0]


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_cell_id():
    grid = examples.load_explicit_structured()

    ind = grid.cell_id((3, 4, 0))
    assert np.issubdtype(ind, np.integer)
    assert ind == 19

    ind = grid.cell_id([(3, 4, 0), (3, 2, 1), (1, 0, 2), (2, 3, 2)])
    assert isinstance(ind, np.ndarray)
    assert np.issubdtype(ind.dtype, np.integer)
    assert np.array_equal(ind, [19, 31, 41, 54])


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
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


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
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


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_compute_connectivity():
    connectivity = np.asarray('''
    42 43 43 41 46 47 47 45 46 47 47 45 46 47 47 45 38 39 39 37 58 59 59 57
    62 63 63 61 62 63 63 61 62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61
    62 63 63 61 62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61 62 63 63 61
    62 63 63 61 54 55 55 53 58 59 59 57 62 63 63 61 62 63 63 61 62 63 63 61
    54 55 55 53 26 27 27 25 30 31 31 29 30 31 31 29 30 31 31 29 22 23 23 21
    '''.split(), dtype=int)

    grid = examples.load_explicit_structured()
    assert 'ConnectivityFlags' not in grid.cell_arrays

    copy = grid.compute_connectivity(inplace=False)
    assert isinstance(copy, pyvista.ExplicitStructuredGrid)
    assert 'ConnectivityFlags' in copy.cell_arrays
    assert 'ConnectivityFlags' not in grid.cell_arrays
    assert np.array_equal(copy.cell_arrays['ConnectivityFlags'], connectivity)

    out = grid.compute_connectivity(inplace=True)
    assert out is grid
    assert 'ConnectivityFlags' in grid.cell_arrays
    assert np.array_equal(grid.cell_arrays['ConnectivityFlags'], connectivity)


@pytest.mark.skipif(not VTK9, reason='VTK 9 or higher is required')
def test_ExplicitStructuredGrid_compute_connections():
    connections = np.asarray('''
    3 4 4 3 4 5 5 4 4 5 5 4 4 5 5 4 3 4 4 3 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4
    5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4 5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6
    6 5 4 5 5 4 4 5 5 4 5 6 6 5 5 6 6 5 5 6 6 5 4 5 5 4 3 4 4 3 4 5 5 4 4 5 5
    4 4 5 5 4 3 4 4 3
    '''.split(), dtype=int)

    grid = examples.load_explicit_structured()
    assert 'number_of_connections' not in grid.cell_arrays

    copy = grid.compute_connections(inplace=False)
    assert isinstance(copy, pyvista.ExplicitStructuredGrid)
    assert 'number_of_connections' in copy.cell_arrays
    assert 'number_of_connections' not in grid.cell_arrays
    assert np.array_equal(copy.cell_arrays['number_of_connections'],
                          connections)

    out = grid.compute_connections(inplace=True)
    assert out is grid
    assert 'number_of_connections' in grid.cell_arrays
    assert np.array_equal(grid.cell_arrays['number_of_connections'],
                          connections)
