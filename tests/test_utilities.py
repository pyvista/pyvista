""" test pyvista.utilities """
import os

import numpy as np
import pytest

import pyvista
from pyvista import examples as ex
from pyvista import utilities
from pyvista import fileio

# Only set this here just the once.
pyvista.set_error_output_file(os.path.join(os.path.dirname(__file__), 'ERROR_OUTPUT.txt'))


def test_createvectorpolydata_error():
    orig = np.random.random((3, 1))
    vec = np.random.random((3, 1))
    with pytest.raises(Exception):
        utilities.vector_poly_data(orig, vec)


def test_createvectorpolydata_1D():
    orig = np.random.random(3)
    vec = np.random.random(3)
    vdata = utilities.vector_poly_data(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.point_arrays['vectors'])


def test_createvectorpolydata():
    orig = np.random.random((100, 3))
    vec = np.random.random((100, 3))
    vdata = utilities.vector_poly_data(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.point_arrays['vectors'])


def test_read(tmpdir):
    fnames = (ex.antfile, ex.planefile, ex.hexbeamfile, ex.spherefile,
              ex.uniformfile, ex.rectfile)
    types = (pyvista.PolyData, pyvista.PolyData, pyvista.UnstructuredGrid,
             pyvista.PolyData, pyvista.UniformGrid, pyvista.RectilinearGrid)
    for i, filename in enumerate(fnames):
        obj = fileio.read(filename)
        assert isinstance(obj, types[i])
    # Now test the standard_reader_routine
    for i, filename in enumerate(fnames):
        # Pass attrs to for the standard_reader_routine to be used
        obj = fileio.read(filename, attrs={'DebugOn': None})
        assert isinstance(obj, types[i])
    # this is also tested for each mesh types init from file tests
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % 'npy'))
    arr = np.random.rand(10, 10)
    np.save(filename, arr)
    with pytest.raises(IOError):
        data = pyvista.read(filename)
    # read non existing file
    with pytest.raises(IOError):
        data = pyvista.read('this_file_totally_does_not_exist.vtk')


def test_get_array():
    grid = pyvista.UnstructuredGrid(ex.hexbeamfile)
    # add array to both point/cell data with same name
    carr = np.random.rand(grid.n_cells)
    grid._add_cell_array(carr, 'test_data')
    parr = np.random.rand(grid.n_points)
    grid._add_point_array(parr, 'test_data')
    # add other data
    oarr = np.random.rand(grid.n_points)
    grid._add_point_array(oarr, 'other')
    farr = np.random.rand(grid.n_points * grid.n_cells)
    grid._add_field_array(farr, 'field_data')
    assert np.allclose(carr, utilities.get_array(grid, 'test_data', preference='cell'))
    assert np.allclose(parr, utilities.get_array(grid, 'test_data', preference='point'))
    assert np.allclose(oarr, utilities.get_array(grid, 'other'))
    assert None == utilities.get_array(grid, 'foo')
    assert utilities.get_array(grid, 'test_data', preference='field') is None
    assert np.allclose(farr, utilities.get_array(grid, 'field_data', preference='field'))




def test_is_inside_bounds():
    data = ex.load_uniform()
    bnds = data.bounds
    assert utilities.is_inside_bounds((0.5, 0.5, 0.5), bnds)
    assert not utilities.is_inside_bounds((12, 5, 5), bnds)
    assert not utilities.is_inside_bounds((5, 12, 5), bnds)
    assert not utilities.is_inside_bounds((5, 5, 12), bnds)
    assert not utilities.is_inside_bounds((12, 12, 12), bnds)


def test_get_sg_image_scraper():
    scraper = pyvista._get_sg_image_scraper()
    assert isinstance(scraper, pyvista.Scraper)
    assert callable(scraper)


def test_voxelize():
    # mesh = examples.load_airplane()
    pass


def test_report():
    report = pyvista.Report()
    assert report is not None


def test_lines_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    poly = pyvista.lines_from_points(points)
    assert poly.n_cells == 2
    assert poly.n_points == 4
    cells = poly.lines
    assert np.allclose(cells[0], [2, 0,1])
    assert np.allclose(cells[1], [2, 2,3])
