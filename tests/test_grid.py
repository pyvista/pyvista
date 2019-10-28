import os

import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples
from pyvista.plotting import system_supports_plotting

beam = pyvista.UnstructuredGrid(examples.hexbeamfile)

# create structured grid
x = np.arange(-10, 10, 2)
y = np.arange(-10, 10, 2)
z = np.arange(-10, 10, 2)
x, y, z = np.meshgrid(x, y, z)
sgrid = pyvista.StructuredGrid(x, y, z)

try:
    test_path = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(test_path, 'test_data')
except:
    test_path = '/home/alex/afrl/python/source/pyvista/tests'


def test_volume():
    assert beam.volume > 0.0


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_struct_example():
    # create and plot structured grid
    grid = examples.load_structured()
    cpos = grid.plot(off_screen=True)  # basic plot
    assert isinstance(cpos, list)

    # Plot mean curvature
    cpos_curv = grid.plot_curvature(off_screen=True)
    assert isinstance(cpos_curv, list)


def test_init_from_structured():
    unstruct_grid = pyvista.UnstructuredGrid(sgrid)
    assert unstruct_grid.points.shape[0] == x.size
    assert np.all(unstruct_grid.celltypes == 12)


def test_init_from_unstructured():
    grid = pyvista.UnstructuredGrid(beam, deep=True)
    grid.points += 1
    assert not np.any(grid.points == beam.points)

def test_init_bad_input():
    with pytest.raises(Exception):
        unstruct_grid = pyvista.UnstructuredGrid(np.array(1))

    with pytest.raises(Exception):
        unstruct_grid = pyvista.UnstructuredGrid(np.array(1),
                                                 np.array(1),
                                                 np.array(1),
                                                 'woa')


def test_init_from_arrays():
    offset = np.array([0, 9], np.int8)
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
    grid = pyvista.UnstructuredGrid(offset, cells, cell_type, points)

    assert grid.n_cells == 2
    assert np.allclose(grid.offset, offset)

def test_surface_indices():
    surf = beam.extract_surface()
    surf_ind = surf.point_arrays['vtkOriginalPointIds']
    assert np.allclose(surf_ind, beam.surface_indices())


def test_extract_edges():
    edges = beam.extract_edges(90)
    assert edges.n_points

    edges = beam.extract_edges(180)
    assert not edges.n_points


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['vtu', 'vtk'])
def test_save(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
    beam.save(filename, binary)

    grid = pyvista.UnstructuredGrid(filename)
    assert grid.cells.shape == beam.cells.shape
    assert grid.points.shape == beam.points.shape

    grid = pyvista.read(filename)
    assert grid.cells.shape == beam.cells.shape
    assert grid.points.shape == beam.points.shape
    assert isinstance(grid, pyvista.UnstructuredGrid)


def test_init_bad_filename():
    filename = os.path.join(test_path, 'test_grid.py')
    with pytest.raises(Exception):
        grid = pyvista.UnstructuredGrid(filename)

    with pytest.raises(Exception):
        grid = pyvista.UnstructuredGrid('not a file')


def test_save_bad_extension():
    with pytest.raises(Exception):
        grid = pyvista.UnstructuredGrid('file.abc')


def test_linear_copy():
    # need a grid with quadratic cells
    lgrid = beam.linear_copy()
    assert np.all(lgrid.celltypes < 20)


def test_extract_cells():
    ind = [1, 2, 3]
    part_beam = beam.extract_cells(ind)
    assert part_beam.n_cells == len(ind)
    assert part_beam.n_points < beam.n_points

    mask = np.zeros(beam.n_cells, np.bool)
    mask[:3] = True
    part_beam = beam.extract_cells(mask)
    assert part_beam.n_cells == len(ind)
    assert part_beam.n_points < beam.n_points


def test_merge():
    grid = beam.copy()
    grid.points[:, 0] += 1
    unmerged = grid.merge(beam, inplace=False, merge_points=False)

    grid.merge(beam, inplace=True, merge_points=True)
    assert grid.n_points > beam.n_points
    assert grid.n_points < unmerged.n_points


def test_merge_not_main():
    grid = beam.copy()
    grid.points[:, 0] += 1
    unmerged = grid.merge(beam, inplace=False, merge_points=False,
                          main_has_priority=False)

    grid.merge(beam, inplace=True, merge_points=True)
    assert grid.n_points > beam.n_points
    assert grid.n_points < unmerged.n_points


def test_merge_list():
    grid_a = beam.copy()
    grid_a.points[:, 0] += 1

    grid_b = beam.copy()
    grid_b.points[:, 1] += 1

    grid_a.merge([beam, grid_b], inplace=True, merge_points=True)
    assert grid_a.n_points > beam.n_points


def test_init_structured():
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 2)
    zrng = np.arange(-10, 10, 2)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    grid = pyvista.StructuredGrid(x, y, z)
    assert np.allclose(sgrid.x, x)
    assert np.allclose(sgrid.y, y)
    assert np.allclose(sgrid.z, z)

    grid_a = pyvista.StructuredGrid(grid)
    assert np.allclose(grid_a.points, grid.points)


def test_invalid_init_structured():
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 2)
    zrng = np.arange(-10, 10, 2)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    z = z[:, :, :2]
    with pytest.raises(Exception):
        grid = pyvista.StructuredGrid(x, y, z)


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['vts', 'vtk'])
def test_save_structured(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
    sgrid.save(filename, binary)

    grid = pyvista.StructuredGrid(filename)
    assert grid.x.shape == sgrid.y.shape
    assert grid.n_cells
    assert grid.points.shape == sgrid.points.shape

    grid = pyvista.read(filename)
    assert grid.x.shape == sgrid.y.shape
    assert grid.n_cells
    assert grid.points.shape == sgrid.points.shape
    assert isinstance(grid, pyvista.StructuredGrid)


def test_load_structured_bad_filename():
    with pytest.raises(Exception):
        pyvista.StructuredGrid('not a file')

    filename = os.path.join(test_path, 'test_grid.py')
    with pytest.raises(Exception):
        grid = pyvista.StructuredGrid(filename)


def test_create_rectilinear_grid_from_specs():
    # 3D example
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 5)
    zrng = np.arange(-10, 10, 1)
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


def test_create_uniform_grid_from_specs():
    # create UniformGrid
    dims = [10, 10, 10]
    grid = pyvista.UniformGrid(dims) # Using default spacing and origin
    assert grid.dimensions == [10, 10, 10]
    assert grid.extent == [0, 9, 0, 9, 0, 9]
    assert grid.origin == [0.0, 0.0, 0.0]
    assert grid.spacing == [1.0, 1.0, 1.0]
    spacing = [2, 1, 5]
    grid = pyvista.UniformGrid(dims, spacing) # Usign default origin
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
@pytest.mark.parametrize('extension', ['vtr', 'vtk'])
def test_save_rectilinear(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
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
@pytest.mark.parametrize('extension', ['vti', 'vtk'])
def test_save_uniform(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
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
    """Test the points mehtods on UniformGrid and inearGrid"""
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1],
                       [1, 1, 1],
                       [0, 1, 1]])
    grid = pyvista.UniformGrid()
    grid.points = points
    assert grid.dimensions == [2, 2, 2]
    assert grid.spacing == [1, 1, 1]
    assert grid.origin == [0., 0., 0.]
    assert np.allclose(np.unique(grid.points, axis=0), np.unique(points, axis=0))
    opts = np.c_[grid.x, grid.y, grid.z]
    assert np.allclose(np.unique(opts, axis=0), np.unique(points, axis=0))
    # Now test rectilinear grid
    del grid
    grid = pyvista.RectilinearGrid()
    grid.points = points
    assert grid.dimensions == [2, 2, 2]
    assert np.allclose(np.unique(grid.points, axis=0), np.unique(points, axis=0))


def test_grid_extract_selection_points():
    grid = pyvista.UnstructuredGrid(sgrid)
    sub_grid = grid.extract_selection_points([0])
    assert sub_grid.n_cells == 1

    sub_grid = grid.extract_selection_points(range(100))
    assert sub_grid.n_cells > 1


def test_gaussian_smooth():
    uniform = examples.load_uniform()
    active = uniform.active_scalar_name
    values = uniform.active_scalar

    uniform = uniform.gaussian_smooth(scalars=active)
    assert uniform.active_scalar_name == active
    assert uniform.active_scalar.shape == values.shape
    assert not np.all(uniform.active_scalar == values)
    values = uniform.active_scalar

    uniform = uniform.gaussian_smooth(radius_factor=5, std_dev=1.3)
    assert uniform.active_scalar_name == active
    assert uniform.active_scalar.shape == values.shape
    assert not np.all(uniform.active_scalar == values)
