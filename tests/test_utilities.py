""" test vtki.utilities """
import os

import numpy as np
import pytest

import vtki
from vtki import examples as ex
from vtki import utilities
from vtki import readers

# Only set this here just the once.
vtki.set_error_output_file(os.path.join(os.path.dirname(__file__), 'ERROR_OUTPUT.txt'))


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
    types = (vtki.PolyData, vtki.PolyData, vtki.UnstructuredGrid,
             vtki.PolyData, vtki.UniformGrid, vtki.RectilinearGrid)
    for i, filename in enumerate(fnames):
        obj = readers.read(filename)
        assert isinstance(obj, types[i])
    # Now test the standard_reader_routine
    for i, filename in enumerate(fnames):
        # Pass attrs to for the standard_reader_routine to be used
        obj = readers.read(filename, attrs={'DebugOn': None})
        assert isinstance(obj, types[i])
    # this is also tested for each mesh types init from file tests
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.%s' % 'npy'))
    arr = np.random.rand(10, 10)
    np.save(filename, arr)
    with pytest.raises(IOError):
        data = vtki.read(filename)


def test_get_scalar():
    grid = vtki.UnstructuredGrid(ex.hexbeamfile)
    # add array to both point/cell data with same name
    carr = np.random.rand(grid.n_cells)
    grid._add_cell_scalar(carr, 'test_data')
    parr = np.random.rand(grid.n_points)
    grid._add_point_scalar(parr, 'test_data')
    oarr = np.random.rand(grid.n_points)
    grid._add_point_scalar(oarr, 'other')
    assert np.allclose(carr, utilities.get_scalar(grid, 'test_data', preference='cell'))
    assert np.allclose(parr, utilities.get_scalar(grid, 'test_data', preference='point'))
    assert np.allclose(oarr, utilities.get_scalar(grid, 'other'))
    assert None == utilities.get_scalar(grid, 'foo')
    # check errors
    with pytest.raises(RuntimeError):
        foo = utilities.get_scalar(grid, 'test_data', preference='field')



def test_is_inside_bounds():
    data = ex.load_uniform()
    bnds = data.bounds
    assert utilities.is_inside_bounds((0.5, 0.5, 0.5), bnds)
    assert not utilities.is_inside_bounds((12, 5, 5), bnds)
    assert not utilities.is_inside_bounds((5, 12, 5), bnds)
    assert not utilities.is_inside_bounds((5, 5, 12), bnds)
    assert not utilities.is_inside_bounds((12, 12, 12), bnds)
