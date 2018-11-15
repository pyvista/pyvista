import os

import vtk
import numpy as np
import pytest

import vtki
from vtki import examples
from vtki.plotting import running_xserver

beam = vtki.UnstructuredGrid(examples.hexbeamfile)

# create structured grid
x = np.arange(-10, 10, 2)
y = np.arange(-10, 10, 2)
z = np.arange(-10, 10, 2)
x, y, z = np.meshgrid(x, y, z)
sgrid = vtki.StructuredGrid(x, y, z)

try:
    test_path = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(test_path, 'test_data')
except:
    test_path = '/home/alex/afrl/python/source/vtki/tests'


def test_volume():
    assert beam.volume > 0.0    


def test_merge():
    beamA = vtki.UnstructuredGrid(examples.hexbeamfile)
    beamB = beamA.copy()
    beamB.points[:, 1] += 1
    beamA.Merge(beamB)


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_struct_example():
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)

    # create and plot structured grid
    grid = vtki.StructuredGrid(x, y, z)
    cpos = grid.plot(off_screen=True)  # basic plot
    assert isinstance(cpos, list)

    # Plot mean curvature
    cpos_curv = grid.plot_curvature(off_screen=True)
    assert isinstance(cpos_curv, list)


def test_init_from_structured():
    unstruct_grid = vtki.UnstructuredGrid(sgrid)
    assert unstruct_grid.points.shape[0] == x.size
    assert np.all(unstruct_grid.celltypes == 12)


def test_init_from_unstructured():
    grid = vtki.UnstructuredGrid(beam, deep=True)
    grid.points += 1
    assert not np.any(grid.points == beam.points)

def test_init_bad_input():
    with pytest.raises(Exception):
        unstruct_grid = vtki.UnstructuredGrid(np.array(1))

    with pytest.raises(Exception):
        unstruct_grid = vtki.UnstructuredGrid(np.array(1),
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
    grid = vtki.UnstructuredGrid(offset, cells, cell_type, points)

    assert grid.number_of_cells == 2
    assert np.allclose(grid.offset, offset)

def test_surface_indices():
    surf = beam.extract_surface()
    surf_ind = surf.point_arrays['vtkOriginalPointIds']
    assert np.allclose(surf_ind, beam.surface_indices())


def test_extract_edges():
    edges = beam.extract_edges(90)
    assert edges.number_of_points

    edges = beam.extract_edges(180)
    assert not edges.number_of_points


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['vtu', 'vtk'])
def test_save(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
    beam.save(filename, binary)

    grid = vtki.UnstructuredGrid(filename)
    assert grid.cells.shape == beam.cells.shape
    assert grid.points.shape == beam.points.shape


def test_init_bad_filename():
    filename = os.path.join(test_path, 'test_grid.py')
    with pytest.raises(Exception):
        grid = vtki.UnstructuredGrid(filename)

    with pytest.raises(Exception):
        grid = vtki.UnstructuredGrid('not a file')


def test_save_bad_extension():
    with pytest.raises(Exception):
        grid = vtki.UnstructuredGrid('file.abc')


def test_linear_copy():
    # need a grid with quadratic cells
    lgrid = beam.linear_copy()
    assert np.all(lgrid.celltypes < 20)


def test_extract_cells():
    ind = [1, 2, 3]
    part_beam = beam.extract_cells(ind)
    assert part_beam.number_of_cells == len(ind)
    assert part_beam.number_of_points < beam.number_of_points

    mask = np.zeros(beam.number_of_cells, np.bool)
    mask[:3] = True
    part_beam = beam.extract_cells(mask)
    assert part_beam.number_of_cells == len(ind)
    assert part_beam.number_of_points < beam.number_of_points    


def test_merge():
    grid = beam.copy()
    grid.points[:, 0] += 1
    unmerged = grid.merge(beam, inplace=False, merge_points=False)

    grid.merge(beam)
    assert grid.number_of_points > beam.number_of_points
    assert grid.number_of_points < unmerged.number_of_points


def test_merge_not_main():
    grid = beam.copy()
    grid.points[:, 0] += 1
    unmerged = grid.merge(beam, inplace=False, merge_points=False,
                          main_has_priority=False)

    grid.merge(beam)
    assert grid.number_of_points > beam.number_of_points
    assert grid.number_of_points < unmerged.number_of_points


def test_merge_list():
    grid_a = beam.copy()
    grid_a.points[:, 0] += 1

    grid_b = beam.copy()
    grid_b.points[:, 1] += 1

    grid_a.merge([beam, grid_b])
    assert grid_a.number_of_points > beam.number_of_points


def test_init_structured():
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 2)
    zrng = np.arange(-10, 10, 2)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    grid = vtki.StructuredGrid(x, y, z)
    assert np.allclose(sgrid.x, x)
    assert np.allclose(sgrid.y, y)
    assert np.allclose(sgrid.z, z)

    grid_a = vtki.StructuredGrid(grid)
    assert np.allclose(grid_a.points, grid.points)


def test_invalid_init_structured():
    xrng = np.arange(-10, 10, 2)
    yrng = np.arange(-10, 10, 2)
    zrng = np.arange(-10, 10, 2)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    z = z[:, :, :2]
    with pytest.raises(Exception):
        grid = vtki.StructuredGrid(x, y, z)


@pytest.mark.parametrize('binary', [True, False])
@pytest.mark.parametrize('extension', ['vts', 'vtk'])
def test_save_structured(extension, binary, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % extension))
    sgrid.save(filename, binary)

    grid = vtki.StructuredGrid(filename)
    assert grid.x.shape == sgrid.y.shape
    assert grid.number_of_cells
    assert grid.points.shape == sgrid.points.shape


def test_load_structured_bad_filename():
    with pytest.raises(Exception):
        vtki.StructuredGrid('not a file')

    filename = os.path.join(test_path, 'test_grid.py')
    with pytest.raises(Exception):
        grid = vtki.StructuredGrid(filename)
