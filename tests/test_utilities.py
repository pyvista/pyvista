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
    assert utilities.get_array(grid, 'foo') is None
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
    mesh = pyvista.PolyData(ex.load_uniform().points)
    vox = pyvista.voxelize(mesh, 0.5)
    assert vox.n_cells


def test_report():
    report = pyvista.Report()
    assert report is not None


def test_line_segments_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    poly = pyvista.line_segments_from_points(points)
    assert poly.n_cells == 2
    assert poly.n_points == 4
    cells = poly.lines
    assert np.allclose(cells[0], [2, 0,1])
    assert np.allclose(cells[1], [2, 2,3])


def test_lines_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    poly = pyvista.lines_from_points(points)
    assert poly.n_cells == 2
    assert poly.n_points == 3
    cells = poly.lines
    assert np.allclose(cells[0], [2, 0,1])
    assert np.allclose(cells[1], [2, 1,2])


def test_grid_from_sph_coords():
    x = np.arange(0.0, 360.0, 40.0)  # longitude
    y = np.arange(0.0, 181.0, 60.0)  # colatitude
    z = [1]  # elevation (radius)
    g = pyvista.grid_from_sph_coords(x, y, z)
    assert g.n_cells == 24
    assert g.n_points == 36
    assert np.allclose(
        g.bounds,
        [
            -0.8137976813493738,
            0.8660254037844387,
            -0.8528685319524434,
            0.8528685319524433,
            -1.0,
            1.0,
        ],
    )
    assert np.allclose(g.points[1], [0.8660254037844386, 0.0, 0.5])
    z = np.linspace(10, 30, 3)
    g = pyvista.grid_from_sph_coords(x, y, z)
    assert g.n_cells == 48
    assert g.n_points == 108
    assert np.allclose(g.points[0], [0.0, 0.0, 10.0])


def test_transform_vectors_sph_to_cart():
    lon = np.arange(0.0, 360.0, 40.0)  # longitude
    lat = np.arange(0.0, 181.0, 60.0)  # colatitude
    lev = [1]  # elevation (radius)
    u, v = np.meshgrid(lon, lat, indexing="ij")
    w = u ** 2 - v ** 2
    uu, vv, ww = pyvista.transform_vectors_sph_to_cart(lon, lat, lev, u, v, w)
    assert np.allclose(
        [uu[-1, -1], vv[-1, -1], ww[-1, -1]],
        [67.80403533828323, 360.8359915416445, -70000.0],
    )
