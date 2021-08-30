""" test pyvista.utilities """
import warnings
import pathlib
import os
import shutil

import numpy as np
import pytest
import unittest.mock as mock

import vtk

import pyvista
from pyvista import examples as ex
from pyvista.utilities import (
    check_valid_vector,
    errors,
    fileio,
    GPUInfo,
    helpers,
    Observer,
    cells,
    transformations
)


def test_createvectorpolydata_error():
    orig = np.random.random((3, 1))
    vec = np.random.random((3, 1))
    with pytest.raises(ValueError):
        helpers.vector_poly_data(orig, vec)


def test_createvectorpolydata_1D():
    orig = np.random.random(3)
    vec = np.random.random(3)
    vdata = helpers.vector_poly_data(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.point_data['vectors'])


def test_createvectorpolydata():
    orig = np.random.random((100, 3))
    vec = np.random.random((100, 3))
    vdata = helpers.vector_poly_data(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.point_data['vectors'])


@pytest.mark.parametrize('use_pathlib', [True, False])
def test_read(tmpdir, use_pathlib):
    fnames = (ex.antfile, ex.planefile, ex.hexbeamfile, ex.spherefile,
              ex.uniformfile, ex.rectfile)
    if use_pathlib:
        fnames = [pathlib.Path(fname) for fname in fnames]
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
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp.npy'))
    arr = np.random.rand(10, 10)
    np.save(filename, arr)
    with pytest.raises(IOError):
        _ = pyvista.read(filename)
    # read non existing file
    with pytest.raises(IOError):
        _ = pyvista.read('this_file_totally_does_not_exist.vtk')
    # Now test reading lists of files as multi blocks
    multi = pyvista.read(fnames)
    assert isinstance(multi, pyvista.MultiBlock)
    assert multi.n_blocks == len(fnames)
    nested = [ex.planefile,
              [ex.hexbeamfile, ex.uniformfile]]

    multi = pyvista.read(nested)
    assert isinstance(multi, pyvista.MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi[1], pyvista.MultiBlock)
    assert multi[1].n_blocks == 2


def test_read_force_ext(tmpdir):
    fnames = (ex.antfile, ex.planefile, ex.hexbeamfile, ex.spherefile,
              ex.uniformfile, ex.rectfile)
    types = (pyvista.PolyData, pyvista.PolyData, pyvista.UnstructuredGrid,
             pyvista.PolyData, pyvista.UniformGrid, pyvista.RectilinearGrid)

    dummy_extension = '.dummy'
    for fname, type in zip(fnames, types):
        root, original_ext = os.path.splitext(fname)
        _, name = os.path.split(root)
        new_fname = tmpdir / name + '.' + dummy_extension
        shutil.copy(fname, new_fname)
        data = fileio.read(new_fname, force_ext=original_ext)
        assert isinstance(data, type)


def test_read_force_ext_wrong_extension(tmpdir):
    # try to read a .vtu file as .vts
    # vtkXMLStructuredGridReader throws a VTK error about the validity of the XML file
    # the returned dataset is empty
    fname = tmpdir / 'airplane.vtu'
    ex.load_airplane().cast_to_unstructured_grid().save(fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fileio.read(fname, force_ext='.vts')
    assert data.n_points == 0

    # try to read a .ply file as .vtm
    # vtkXMLMultiBlockDataReader throws a VTK error about the validity of the XML file
    # the returned dataset is empty
    fname = ex.planefile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fileio.read(fname, force_ext='.vtm')
    assert len(data) == 0


@mock.patch('pyvista.utilities.fileio.standard_reader_routine')
def test_read_legacy(srr_mock):
    srr_mock.return_value = pyvista.read(ex.planefile)
    pyvista.read_legacy('legacy.vtk')
    args, kwargs = srr_mock.call_args
    reader = args[0]
    assert isinstance(reader, vtk.vtkDataSetReader)
    assert reader.GetFileName().endswith('legacy.vtk')

    # check error is raised when no data returned
    srr_mock.reset_mock()
    srr_mock.return_value = None
    with pytest.raises(RuntimeError):
        pyvista.read_legacy('legacy.vtk')


@mock.patch('pyvista.utilities.fileio.read_legacy')
def test_pyvista_read_legacy(read_legacy_mock):
    # check that reading a file with extension .vtk calls `read_legacy`
    # use the globefile as a dummy because pv.read() checks for the existence of the file
    pyvista.read(ex.globefile)
    args, kwargs = read_legacy_mock.call_args
    filename = args[0]
    assert filename == ex.globefile


@mock.patch('pyvista.utilities.fileio.read_exodus')
def test_pyvista_read_exodus(read_exodus_mock):
    # check that reading a file with extension .e calls `read_exodus`
    # use the globefile as a dummy because pv.read() checks for the existence of the file
    pyvista.read(ex.globefile, force_ext='.e')
    args, kwargs = read_exodus_mock.call_args
    filename = args[0]
    assert filename == ex.globefile


@pytest.mark.parametrize('auto_detect', (True, False))
@mock.patch('pyvista.utilities.fileio.standard_reader_routine')
def test_read_plot3d(srr_mock, auto_detect):
    # with grid only
    pyvista.read_plot3d(filename='grid.in', auto_detect=auto_detect)
    srr_mock.assert_called_once()
    args, kwargs = srr_mock.call_args
    reader = args[0]
    assert isinstance(reader, vtk.vtkMultiBlockPLOT3DReader)
    assert reader.GetFileName().endswith('grid.in')
    assert kwargs['filename'] is None
    assert kwargs['attrs'] == {'SetAutoDetectFormat': auto_detect}

    # with grid and q
    srr_mock.reset_mock()
    pyvista.read_plot3d(filename='grid.in', q_filenames='q1.save',
                        auto_detect=auto_detect)
    args, kwargs = srr_mock.call_args
    reader = args[0]
    assert isinstance(reader, vtk.vtkMultiBlockPLOT3DReader)
    assert reader.GetFileName().endswith('grid.in')
    assert args[0].GetQFileName().endswith('q1.save')
    assert kwargs['filename'] is None
    assert kwargs['attrs'] == {'SetAutoDetectFormat': auto_detect}


def test_get_array():
    grid = pyvista.UnstructuredGrid(ex.hexbeamfile)
    # add array to both point/cell data with same name
    carr = np.random.rand(grid.n_cells)
    grid.cell_data.set_array(carr, 'test_data')
    parr = np.random.rand(grid.n_points)
    grid.point_data.set_array(parr, 'test_data')
    # add other data
    oarr = np.random.rand(grid.n_points)
    grid.point_data.set_array(oarr, 'other')
    farr = np.random.rand(grid.n_points * grid.n_cells)
    grid.field_data.set_array(farr, 'field_data')
    assert np.allclose(carr, helpers.get_array(grid, 'test_data', preference='cell'))
    assert np.allclose(parr, helpers.get_array(grid, 'test_data', preference='point'))
    assert np.allclose(oarr, helpers.get_array(grid, 'other'))
    assert helpers.get_array(grid, 'foo') is None
    assert helpers.get_array(grid, 'test_data', preference='field') is None
    assert np.allclose(farr, helpers.get_array(grid, 'field_data', preference='field'))


def test_is_inside_bounds():
    data = ex.load_uniform()
    bnds = data.bounds
    assert helpers.is_inside_bounds((0.5, 0.5, 0.5), bnds)
    assert not helpers.is_inside_bounds((12, 5, 5), bnds)
    assert not helpers.is_inside_bounds((5, 12, 5), bnds)
    assert not helpers.is_inside_bounds((5, 5, 12), bnds)
    assert not helpers.is_inside_bounds((12, 12, 12), bnds)


def test_get_sg_image_scraper():
    scraper = pyvista._get_sg_image_scraper()
    assert isinstance(scraper, pyvista.Scraper)
    assert callable(scraper)


def test_voxelize():
    mesh = pyvista.PolyData(ex.load_uniform().points)
    vox = pyvista.voxelize(mesh, 0.5)
    assert vox.n_cells

def test_voxelize_non_uniform_desnity():
    mesh = pyvista.PolyData(ex.load_uniform().points)
    vox = pyvista.voxelize(mesh, [0.5, 0.3, 0.2])
    assert vox.n_cells

def test_voxelize_throws_when_density_is_not_length_3():
    with pytest.raises(ValueError) as e:
        mesh = pyvista.PolyData(ex.load_uniform().points)
        vox = pyvista.voxelize(mesh, [0.5, 0.3])
    assert "not enough values to unpack" in str(e.value)

def test_report():
    report = pyvista.Report(gpu=True)
    assert report is not None
    report = pyvista.Report(gpu=False)
    assert report is not None


def test_line_segments_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    poly = pyvista.line_segments_from_points(points)
    assert poly.n_cells == 2
    assert poly.n_points == 4
    cells = poly.lines
    assert np.allclose(cells[:3], [2, 0, 1])
    assert np.allclose(cells[3:], [2, 2, 3])


def test_lines_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    poly = pyvista.lines_from_points(points)
    assert poly.n_cells == 2
    assert poly.n_points == 3
    cells = poly.lines
    assert np.allclose(cells[:3], [2, 0, 1])
    assert np.allclose(cells[3:], [2, 1, 2])


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


def test_vtkmatrix_to_from_array():
    rng = np.random.default_rng()
    array3x3 = rng.integers(0, 10, size=(3, 3))
    matrix = pyvista.vtkmatrix_from_array(array3x3)
    assert isinstance(matrix, vtk.vtkMatrix3x3)
    for i in range(3):
        for j in range(3):
            assert matrix.GetElement(i, j) == array3x3[i, j]

    array = pyvista.array_from_vtkmatrix(matrix)
    assert isinstance(array, np.ndarray)
    assert array.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            assert array[i, j] == matrix.GetElement(i, j)

    array4x4 = rng.integers(0, 10, size=(4, 4))
    matrix = pyvista.vtkmatrix_from_array(array4x4)
    assert isinstance(matrix, vtk.vtkMatrix4x4)
    for i in range(4):
        for j in range(4):
            assert matrix.GetElement(i, j) == array4x4[i, j]

    array = pyvista.array_from_vtkmatrix(matrix)
    assert isinstance(array, np.ndarray)
    assert array.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            assert array[i, j] == matrix.GetElement(i, j)

    # invalid cases
    with pytest.raises(ValueError):
        matrix = pyvista.vtkmatrix_from_array(np.arange(3 * 4).reshape(3, 4))
    with pytest.raises(TypeError):
        invalid = vtk.vtkTransform()
        array = pyvista.array_from_vtkmatrix(invalid)


def test_assert_empty_kwargs():
    kwargs = {}
    assert errors.assert_empty_kwargs(**kwargs)
    with pytest.raises(TypeError):
        kwargs = {"foo":6}
        errors.assert_empty_kwargs(**kwargs)
    with pytest.raises(TypeError):
        kwargs = {"foo":6, "goo":"bad"}
        errors.assert_empty_kwargs(**kwargs)


def test_convert_id_list():
    ids = np.array([4, 5, 8])
    id_list = vtk.vtkIdList()
    id_list.SetNumberOfIds(len(ids))
    for i, v in enumerate(ids):
        id_list.SetId(i, v)
    converted = helpers.vtk_id_list_to_array(id_list)
    assert np.allclose(converted, ids)


def test_progress_monitor():
    mesh = pyvista.Sphere()
    ugrid = mesh.delaunay_3d(progress_bar=True)
    assert isinstance(ugrid, pyvista.UnstructuredGrid)


def test_observer():
    msg = "KIND: In PATH, line 0\nfoo (ADDRESS): ALERT"
    obs = Observer()
    ret = obs.parse_message("foo")
    assert ret[3] == "foo"
    ret = obs.parse_message(msg)
    assert ret[3] == "ALERT"
    for kind in ["WARNING", "ERROR"]:
        obs.log_message(kind, "foo")
    obs(obj=None, event=None, message=msg)
    assert obs.has_event_occurred()
    assert obs.get_message() == "ALERT"
    assert obs.get_message(etc=True) == msg

    alg = vtk.vtkSphereSource()
    alg.GetExecutive()
    obs.observe(alg)
    with pytest.raises(RuntimeError, match="algorithm"):
        obs.observe(alg)


def test_gpuinfo():
    gpuinfo = GPUInfo()
    _repr = gpuinfo.__repr__()
    _repr_html = gpuinfo._repr_html_()
    assert isinstance(_repr, str) and len(_repr) > 1
    assert isinstance(_repr_html, str) and len(_repr_html) > 1

    # test corrupted internal infos
    gpuinfo._gpu_info = 'foo'
    for func_name in ['renderer', 'version', 'vendor']:
        with pytest.raises(RuntimeError, match=func_name):
            getattr(gpuinfo, func_name)()


def test_check_valid_vector():
    with pytest.raises(TypeError, match="length three"):
        check_valid_vector([0, 1])
    check_valid_vector([0, 1, 2])


def test_cells_dict_utils():

    # No pyvista object
    with pytest.raises(ValueError):
        cells.get_mixed_cells(None)

    with pytest.raises(ValueError):
        cells.get_mixed_cells(np.zeros(shape=[3, 3]))

    cells_arr = np.array([3, 0, 1, 2, 3, 3, 4, 5])
    cells_types = np.array([vtk.VTK_TRIANGLE] * 2)

    assert np.all(cells.generate_cell_offsets(cells_arr, cells_types)
                  == cells.generate_cell_offsets(cells_arr, cells_types))

    # Non-integer type
    with pytest.raises(ValueError):
        cells.generate_cell_offsets(cells_arr, cells_types.astype(np.float32))

    with pytest.raises(ValueError):
        cells.generate_cell_offsets_loop(cells_arr, cells_types.astype(np.float32))

    # Inconsistency of cell array lengths
    with pytest.raises(ValueError):
        cells.generate_cell_offsets(np.array(cells_arr.tolist() + [6]), cells_types)

    with pytest.raises(ValueError):
        cells.generate_cell_offsets_loop(np.array(cells_arr.tolist() + [6]), cells_types)

    with pytest.raises(ValueError):
        cells.generate_cell_offsets(cells_arr, np.array(cells_types.tolist() + [vtk.VTK_TRIANGLE]))

    with pytest.raises(ValueError):
        cells.generate_cell_offsets_loop(cells_arr, np.array(cells_types.tolist() + [vtk.VTK_TRIANGLE]))

    # Unknown cell type
    np.all(cells.generate_cell_offsets(cells_arr, cells_types) ==
           cells.generate_cell_offsets(cells_arr, np.array([255, 255])))


def test_apply_transformation_to_points():
    mesh = ex.load_airplane()
    points = mesh.points
    points_orig = points.copy()

    # identity 3 x 3
    tf = np.eye(3)
    points_new = transformations.apply_transformation_to_points(tf, points, inplace=False)
    assert points_new == pytest.approx(points)

    # identity 4 x 4
    tf = np.eye(4)
    points_new = transformations.apply_transformation_to_points(tf, points, inplace=False)
    assert points_new == pytest.approx(points)

    # scale in-place
    tf = np.eye(4) * 2
    tf[3, 3] = 1
    r = transformations.apply_transformation_to_points(tf, points, inplace=True)
    assert r is None
    assert mesh.points == pytest.approx(2 * points_orig)


def _generate_vtk_err():
    """Simple operation which generates a VTK error."""
    x, y, z = np.meshgrid(
        np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5)
    )
    mesh = pyvista.StructuredGrid(x, y, z)
    x2, y2, z2 = np.meshgrid(
        np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5)
    )
    mesh2 = pyvista.StructuredGrid(x2, y2, z2)

    alg = vtk.vtkStreamTracer()
    obs = pyvista.Observer()
    obs.observe(alg)
    alg.SetInputDataObject(mesh)
    alg.SetSourceData(mesh2)
    alg.Update()


def test_vtk_error_catcher():
    # raise_errors: False
    error_catcher = pyvista.utilities.errors.VtkErrorCatcher()
    with error_catcher:
        _generate_vtk_err()
        _generate_vtk_err()
    assert len(error_catcher.events) == 2

    # raise_errors: False, no error
    error_catcher = pyvista.utilities.errors.VtkErrorCatcher()
    with error_catcher:
        pass

    # raise_errors: True
    error_catcher = pyvista.utilities.errors.VtkErrorCatcher(raise_errors=True)
    with pytest.raises(RuntimeError):
        with error_catcher:
            _generate_vtk_err()
    assert len(error_catcher.events) == 1

    # raise_errors: True, no error
    error_catcher = pyvista.utilities.errors.VtkErrorCatcher(raise_errors=True)
    with error_catcher:
        pass


def test_axis_angle_rotation():
    # rotate cube corners around body diagonal
    points = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    axis = [1, 1, 1]

    # no-op case
    angle = 360
    trans = transformations.axis_angle_rotation(axis, angle)
    actual = transformations.apply_transformation_to_points(trans, points)
    assert np.array_equal(actual, points)

    # default origin
    angle = np.radians(120)
    expected = points[[1, 2, 0], :]
    trans = transformations.axis_angle_rotation(axis, angle, deg=False)
    actual = transformations.apply_transformation_to_points(trans, points)
    assert np.allclose(actual, expected)

    # non-default origin
    p0 = [-2, -3, 4]
    points += p0
    expected += p0
    trans = transformations.axis_angle_rotation(axis, angle, point=p0, deg=False)
    actual = transformations.apply_transformation_to_points(trans, points)
    assert np.allclose(actual, expected)

    # invalid cases
    with pytest.raises(ValueError):
        transformations.axis_angle_rotation([1, 0, 0, 0], angle)
    with pytest.raises(ValueError):
        transformations.axis_angle_rotation(axis, angle, point=[1, 0, 0, 0])
    with pytest.raises(ValueError):
        transformations.axis_angle_rotation([0, 0, 0], angle)


def test_reflection():
    # reflect points of a square across a diagonal
    points = np.array([
        [ 1,  1, 0],
        [-1,  1, 0],
        [-1, -1, 0],
        [ 1, -1, 0],
    ])
    normal = [1, 1, 0]

    # default origin
    expected = points[[2, 1, 0, 3], :]
    trans = transformations.reflection(normal)
    actual = transformations.apply_transformation_to_points(trans, points)
    assert np.allclose(actual, expected)

    # non-default origin
    p0 = [1, 1, 0]
    expected += 2 * np.array(p0)
    trans = transformations.reflection(normal, point=p0)
    actual = transformations.apply_transformation_to_points(trans, points)
    assert np.allclose(actual, expected)

    # invalid cases
    with pytest.raises(ValueError):
        transformations.reflection([1, 0, 0, 0])
    with pytest.raises(ValueError):
        transformations.reflection(normal, point=[1, 0, 0, 0])
    with pytest.raises(ValueError):
        transformations.reflection([0, 0, 0])
