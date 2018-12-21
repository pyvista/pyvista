""" test vtki.utilities """
import pytest
import numpy as np
import vtki
from vtki import utilities
from vtki import examples as ex


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


def test_read():
    fnames = (ex.antfile, ex.planefile, ex.hexbeamfile, ex.spherefile,
              ex.uniformfile, ex.rectfile)
    types = (vtki.PolyData, vtki.PolyData, vtki.UnstructuredGrid,
             vtki.PolyData, vtki.UniformGrid, vtki.RectilinearGrid)
    for i, filename in enumerate(fnames):
        obj = utilities.read(filename)
        assert isinstance(obj, types[i])


def test_get_scalar():
    grid = vtki.UnstructuredGrid(ex.hexbeamfile)
    # add array to both point/cell data with same name
    carr = np.random.rand(grid.number_of_cells)
    grid._add_cell_scalar(carr, 'test_data')
    parr = np.random.rand(grid.number_of_points)
    grid._add_point_scalar(parr, 'test_data')
    oarr = np.random.rand(grid.number_of_points)
    grid._add_point_scalar(oarr, 'other')
    assert np.allclose(carr, utilities.get_scalar(grid, 'test_data', preference='cell'))
    assert np.allclose(parr, utilities.get_scalar(grid, 'test_data', preference='point'))
    assert np.allclose(oarr, utilities.get_scalar(grid, 'other'))
    assert None == utilities.get_scalar(grid, 'foo')
