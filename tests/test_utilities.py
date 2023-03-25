""" test pyvista.utilities """
import itertools
import os
import pathlib
import pickle
import shutil
import unittest.mock as mock
import warnings

import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples as ex
from pyvista.plotting import system_supports_plotting
from pyvista.utilities import (
    GPUInfo,
    Observer,
    cells,
    check_valid_vector,
    errors,
    fileio,
    get_ext,
    helpers,
    transformations,
)
from pyvista.utilities.docs import linkcode_resolve
from pyvista.utilities.misc import PyVistaDeprecationWarning, has_duplicates, raise_has_duplicates

skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)


def test_version():
    assert "major" in str(pyvista.vtk_version_info)
    ver = vtk.vtkVersion()
    assert ver.GetVTKMajorVersion() == pyvista.vtk_version_info.major
    assert ver.GetVTKMinorVersion() == pyvista.vtk_version_info.minor
    assert ver.GetVTKBuildVersion() == pyvista.vtk_version_info.micro
    ver_tup = (
        ver.GetVTKMajorVersion(),
        ver.GetVTKMinorVersion(),
        ver.GetVTKBuildVersion(),
    )
    assert ver_tup == pyvista.vtk_version_info
    assert pyvista.vtk_version_info >= (0, 0, 0)


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


@pytest.mark.parametrize(
    'path, target_ext',
    [
        ("/data/mesh.stl", ".stl"),
        ("/data/image.nii.gz", '.nii.gz'),
        ("/data/other.gz", ".gz"),
    ],
)
def test_get_ext(path, target_ext):
    ext = get_ext(path)
    assert ext == target_ext


@pytest.mark.parametrize('use_pathlib', [True, False])
def test_read(tmpdir, use_pathlib):
    fnames = (ex.antfile, ex.planefile, ex.hexbeamfile, ex.spherefile, ex.uniformfile, ex.rectfile)
    if use_pathlib:
        fnames = [pathlib.Path(fname) for fname in fnames]
    types = (
        pyvista.PolyData,
        pyvista.PolyData,
        pyvista.UnstructuredGrid,
        pyvista.PolyData,
        pyvista.UniformGrid,
        pyvista.RectilinearGrid,
    )
    for i, filename in enumerate(fnames):
        obj = fileio.read(filename)
        assert isinstance(obj, types[i])
    # Now test the standard_reader_routine
    for i, filename in enumerate(fnames):
        # Pass attrs to for the standard_reader_routine to be used
        with pytest.warns(PyVistaDeprecationWarning):
            obj = fileio.read(filename, attrs={'DebugOn': None})
        assert isinstance(obj, types[i])
    # this is also tested for each mesh types init from file tests
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.npy'))
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
    nested = [ex.planefile, [ex.hexbeamfile, ex.uniformfile]]

    multi = pyvista.read(nested)
    assert isinstance(multi, pyvista.MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi[1], pyvista.MultiBlock)
    assert multi[1].n_blocks == 2


def test_read_force_ext(tmpdir):
    fnames = (ex.antfile, ex.planefile, ex.hexbeamfile, ex.spherefile, ex.uniformfile, ex.rectfile)
    types = (
        pyvista.PolyData,
        pyvista.PolyData,
        pyvista.UnstructuredGrid,
        pyvista.PolyData,
        pyvista.UniformGrid,
        pyvista.RectilinearGrid,
    )

    dummy_extension = '.dummy'
    for fname, type in zip(fnames, types):
        root, original_ext = os.path.splitext(fname)
        _, name = os.path.split(root)
        new_fname = tmpdir / name + '.' + dummy_extension
        shutil.copy(fname, new_fname)
        data = fileio.read(new_fname, force_ext=original_ext)
        assert isinstance(data, type)


@mock.patch('pyvista.BaseReader.read')
@mock.patch('pyvista.BaseReader.reader')
def test_read_attrs(mock_reader, mock_read):
    """Test passing attrs in read."""
    with pytest.warns(PyVistaDeprecationWarning):
        pyvista.read(ex.antfile, attrs={'test': 'test_arg'})
    mock_reader.test.assert_called_once_with('test_arg')

    mock_reader.reset_mock()
    with pytest.warns(PyVistaDeprecationWarning):
        pyvista.read(ex.antfile, attrs={'test': ['test_arg1', 'test_arg2']})
    mock_reader.test.assert_called_once_with('test_arg1', 'test_arg2')


@mock.patch('pyvista.BaseReader.read')
@mock.patch('pyvista.BaseReader.reader')
@mock.patch('pyvista.BaseReader.show_progress')
def test_read_progress_bar(mock_show_progress, mock_reader, mock_read):
    """Test passing attrs in read."""
    pyvista.read(ex.antfile, progress_bar=True)
    mock_show_progress.assert_called_once()


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

    fname = ex.planefile
    with pytest.raises(IOError):
        fileio.read(fname, force_ext='.not_supported')


@mock.patch('pyvista.utilities.fileio.read')
def test_read_legacy(read_mock):
    with pytest.warns(PyVistaDeprecationWarning):
        pyvista.read_legacy(ex.globefile, progress_bar=False)
    read_mock.assert_called_once_with(ex.globefile, progress_bar=False)


@mock.patch('pyvista.utilities.fileio.read_exodus')
def test_pyvista_read_exodus(read_exodus_mock):
    # check that reading a file with extension .e calls `read_exodus`
    # use the globefile as a dummy because pv.read() checks for the existence of the file
    pyvista.read(ex.globefile, force_ext='.e')
    args, kwargs = read_exodus_mock.call_args
    filename = args[0]
    assert filename == ex.globefile


@pytest.mark.parametrize('auto_detect', (True, False))
@mock.patch('pyvista.utilities.reader.BaseReader.read')
@mock.patch('pyvista.utilities.reader.BaseReader.path')
def test_read_plot3d(path_mock, read_mock, auto_detect):
    # with grid only
    with pytest.warns(PyVistaDeprecationWarning):
        pyvista.read_plot3d(filename='grid.in', auto_detect=auto_detect)
    read_mock.assert_called_once()

    # with grid and q
    read_mock.reset_mock()
    with pytest.warns(PyVistaDeprecationWarning):
        pyvista.read_plot3d(filename='grid.in', q_filenames='q1.save', auto_detect=auto_detect)
    read_mock.assert_called_once()


def test_get_array_cell(hexbeam):
    carr = np.random.rand(hexbeam.n_cells)
    hexbeam.cell_data.set_array(carr, 'test_data')

    data = helpers.get_array(hexbeam, 'test_data', preference='cell')
    assert np.allclose(carr, data)


def test_get_array_point(hexbeam):
    parr = np.random.rand(hexbeam.n_points)
    hexbeam.point_data.set_array(parr, 'test_data')

    data = helpers.get_array(hexbeam, 'test_data', preference='point')
    assert np.allclose(parr, data)

    oarr = np.random.rand(hexbeam.n_points)
    hexbeam.point_data.set_array(oarr, 'other')

    data = helpers.get_array(hexbeam, 'other')
    assert np.allclose(oarr, data)


def test_get_array_field(hexbeam):
    hexbeam.clear_data()
    # no preference
    farr = np.random.rand(hexbeam.n_points * hexbeam.n_cells)
    hexbeam.field_data.set_array(farr, 'data')
    data = helpers.get_array(hexbeam, 'data')
    assert np.allclose(farr, data)

    # preference and multiple data
    hexbeam.point_data.set_array(np.random.rand(hexbeam.n_points), 'data')

    data = helpers.get_array(hexbeam, 'data', preference='field')
    assert np.allclose(farr, data)


def test_get_array_error(hexbeam):
    parr = np.random.rand(hexbeam.n_points)
    hexbeam.point_data.set_array(parr, 'test_data')

    # invalid inputs
    with pytest.raises(TypeError):
        helpers.get_array(hexbeam, 'test_data', preference={'invalid'})
    with pytest.raises(ValueError):
        helpers.get_array(hexbeam, 'test_data', preference='invalid')
    with pytest.raises(ValueError, match='`preference` must be'):
        helpers.get_array(hexbeam, 'test_data', preference='row')


def test_get_array_none(hexbeam):
    arr = helpers.get_array(hexbeam, 'foo')
    assert arr is None


def get_array_vtk(hexbeam):
    # test raw VTK input
    grid_vtk = vtk.vtkUnstructuredGrid()
    grid_vtk.DeepCopy(hexbeam)
    helpers.get_array(grid_vtk, 'test_data')
    helpers.get_array(grid_vtk, 'foo')


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


def test_voxelize(uniform):
    vox = pyvista.voxelize(uniform, 0.5)
    assert vox.n_cells


def test_voxelize_non_uniform_density(uniform):
    vox = pyvista.voxelize(uniform, [0.5, 0.3, 0.2])
    assert vox.n_cells
    vox = pyvista.voxelize(uniform, np.array([0.5, 0.3, 0.2]))
    assert vox.n_cells


def test_voxelize_invalid_density(rectilinear):
    # test error when density is not length-3
    with pytest.raises(ValueError, match='not enough values to unpack'):
        pyvista.voxelize(rectilinear, [0.5, 0.3])
    # test error when density is not an array-like
    with pytest.raises(TypeError, match='expected number or array-like'):
        pyvista.voxelize(rectilinear, {0.5, 0.3})


def test_voxelize_throws_point_cloud(hexbeam):
    with pytest.raises(ValueError, match='must have faces'):
        mesh = pyvista.PolyData(hexbeam.points)
        pyvista.voxelize(mesh)


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
    w = u**2 - v**2
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
        kwargs = {"foo": 6}
        errors.assert_empty_kwargs(**kwargs)
    with pytest.raises(TypeError):
        kwargs = {"foo": 6, "goo": "bad"}
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


@skip_no_plotting
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
    with pytest.raises(ValueError, match="length three"):
        check_valid_vector([0, 1])
    check_valid_vector([0, 1, 2])


def test_cells_dict_utils():
    # No pyvista object
    with pytest.raises(ValueError):
        cells.get_mixed_cells(None)

    with pytest.raises(ValueError):
        cells.get_mixed_cells(np.zeros(shape=[3, 3]))


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
    x, y, z = np.meshgrid(np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5))
    mesh = pyvista.StructuredGrid(x, y, z)
    x2, y2, z2 = np.meshgrid(np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5))
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
    points = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
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
    points = np.array(
        [
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
        ]
    )
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


def test_merge(sphere, cube, datasets):
    with pytest.raises(TypeError, match="Expected a sequence"):
        pyvista.merge(None)

    with pytest.raises(ValueError, match="Expected at least one"):
        pyvista.merge([])

    with pytest.raises(TypeError, match="Expected pyvista.DataSet"):
        pyvista.merge([None, sphere])

    # check polydata
    merged_poly = pyvista.merge([sphere, cube])
    assert isinstance(merged_poly, pyvista.PolyData)
    assert merged_poly.n_points == sphere.n_points + cube.n_points

    merged = pyvista.merge([sphere, sphere], merge_points=True)
    assert merged.n_points == sphere.n_points

    merged = pyvista.merge([sphere, sphere], merge_points=False)
    assert merged.n_points == sphere.n_points * 2

    # check unstructured
    merged_ugrid = pyvista.merge(datasets, merge_points=False)
    assert isinstance(merged_ugrid, pyvista.UnstructuredGrid)
    assert merged_ugrid.n_points == sum([ds.n_points for ds in datasets])
    # check main has priority
    sphere_a = sphere.copy()
    sphere_b = sphere.copy()
    sphere_a['data'] = np.zeros(sphere_a.n_points)
    sphere_b['data'] = np.ones(sphere_a.n_points)

    merged = pyvista.merge(
        [sphere_a, sphere_b],
        merge_points=True,
        main_has_priority=False,
    )
    assert np.allclose(merged['data'], 1)

    merged = pyvista.merge(
        [sphere_a, sphere_b],
        merge_points=True,
        main_has_priority=True,
    )
    assert np.allclose(merged['data'], 0)


def test_color():
    name, name2 = "blue", "b"
    i_rgba, f_rgba = (0, 0, 255, 255), (0.0, 0.0, 1.0, 1.0)
    h = "0000ffff"
    i_opacity, f_opacity, h_opacity = 153, 0.6, "99"
    invalid_colors = (
        (300, 0, 0),
        (0, -10, 0),
        (0, 0, 1.5),
        (-0.5, 0, 0),
        (0, 0),
        "#hh0000",
        "invalid_name",
        {"invalid_name": 100},
    )
    invalid_opacities = (275, -50, 2.4, -1.2, "#zz")
    i_types = (int, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
    f_types = (float, np.float16, np.float32, np.float64)
    h_prefixes = ("", "0x", "#")
    assert pyvista.Color(name) == i_rgba
    assert pyvista.Color(name2) == i_rgba
    # Check integer types
    for i_type in i_types:
        i_color = [i_type(c) for c in i_rgba]
        # Check list, tuple and numpy array
        assert pyvista.Color(i_color) == i_rgba
        assert pyvista.Color(tuple(i_color)) == i_rgba
        assert pyvista.Color(np.asarray(i_color, dtype=i_type)) == i_rgba
    # Check float types
    for f_type in f_types:
        f_color = [f_type(c) for c in f_rgba]
        # Check list, tuple and numpy array
        assert pyvista.Color(f_color) == i_rgba
        assert pyvista.Color(tuple(f_color)) == i_rgba
        assert pyvista.Color(np.asarray(f_color, dtype=f_type)) == i_rgba
    # Check hex
    for h_prefix in h_prefixes:
        assert pyvista.Color(h_prefix + h) == i_rgba
    # Check dict
    for channels in itertools.product(*pyvista.Color.CHANNEL_NAMES):
        dct = dict(zip(channels, i_rgba))
        assert pyvista.Color(dct) == i_rgba
    # Check opacity
    for opacity in (i_opacity, f_opacity, h_opacity):
        # No opacity in color provided => use opacity
        assert pyvista.Color(name, opacity) == (*i_rgba[:3], i_opacity)
        # Opacity in color provided => overwrite using opacity
        assert pyvista.Color(i_rgba, opacity) == (*i_rgba[:3], i_opacity)
    # Check default_opacity
    for opacity in (i_opacity, f_opacity, h_opacity):
        # No opacity in color provided => use default_opacity
        assert pyvista.Color(name, default_opacity=opacity) == (*i_rgba[:3], i_opacity)
        # Opacity in color provided => keep that opacity
        assert pyvista.Color(i_rgba, default_opacity=opacity) == i_rgba
    # Check default_color
    assert pyvista.Color(None, default_color=name) == i_rgba
    # Check invalid colors and opacities
    for invalid_color in invalid_colors:
        with pytest.raises(ValueError):
            pyvista.Color(invalid_color)
    for invalid_opacity in invalid_opacities:
        with pytest.raises(ValueError):
            pyvista.Color('b', invalid_opacity)
    # Check hex and name getters
    assert pyvista.Color(name).hex_rgba == f'#{h}'
    assert pyvista.Color(name).hex_rgb == f'#{h[:-2]}'
    assert pyvista.Color('b').name == 'blue'
    # Check sRGB conversion
    assert pyvista.Color('gray', 0.5).linear_to_srgb() == '#bcbcbcbc'
    assert pyvista.Color('#bcbcbcbc').srgb_to_linear() == '#80808080'
    # Check iteration and indexing
    c = pyvista.Color(i_rgba)
    assert all(ci == fi for ci, fi in zip(c, f_rgba))
    for i, cnames in enumerate(pyvista.Color.CHANNEL_NAMES):
        assert c[i] == f_rgba[i]
        assert all(c[i] == c[cname] for cname in cnames)
    assert c[-1] == f_rgba[-1]
    assert c[1:3] == f_rgba[1:3]
    with pytest.raises(TypeError):
        c[None]  # Invalid index type
    with pytest.raises(ValueError):
        c["invalid_name"]  # Invalid string index
    with pytest.raises(IndexError):
        c[4]  # Invalid integer index


def test_color_opacity():
    color = pyvista.Color(opacity=0.5)
    assert color.opacity == 128


def test_convert_array():
    arr = np.arange(4).astype('O')
    arr2 = pyvista.utilities.convert_array(arr, array_type=np.dtype('O'))
    assert arr2.GetNumberOfValues() == 4

    # https://github.com/pyvista/pyvista/issues/2370
    arr3 = pyvista.utilities.convert_array(
        pickle.loads(pickle.dumps(np.arange(4).astype('O'))), array_type=np.dtype('O')
    )
    assert arr3.GetNumberOfValues() == 4

    # check lists work
    my_list = [1, 2, 3]
    arr4 = pyvista.utilities.convert_array(my_list)
    assert arr4.GetNumberOfValues() == len(my_list)


def test_has_duplicates():
    assert not has_duplicates(np.arange(100))
    assert has_duplicates(np.array([0, 1, 2, 2]))
    assert has_duplicates(np.array([[0, 1, 2], [0, 1, 2]]))

    with pytest.raises(ValueError):
        raise_has_duplicates(np.array([0, 1, 2, 2]))


def test_copy_vtk_array():
    with pytest.raises(TypeError, match='Invalid type'):
        pyvista.utilities.misc.copy_vtk_array([1, 2, 3])

    value_0 = 10
    value_1 = 10
    arr = vtk.vtkFloatArray()
    arr.SetNumberOfValues(2)
    arr.SetValue(0, value_0)
    arr.SetValue(1, value_1)
    arr_copy = pyvista.utilities.misc.copy_vtk_array(arr, deep=True)
    assert arr_copy.GetNumberOfValues()
    assert value_0 == arr_copy.GetValue(0)

    arr_copy_shallow = pyvista.utilities.misc.copy_vtk_array(arr, deep=False)
    new_value = 5
    arr.SetValue(1, new_value)
    assert value_1 == arr_copy.GetValue(1)
    assert new_value == arr_copy_shallow.GetValue(1)


def test_cartesian_to_spherical():
    def polar2cart(r, theta, phi):
        return np.vstack(
            (r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta))
        ).T

    points = np.random.random((1000, 3))
    x, y, z = points.T
    r, theta, phi = pyvista.cartesian_to_spherical(x, y, z)
    assert np.allclose(polar2cart(r, theta, phi), points)


def test_set_pickle_format():
    pyvista.set_pickle_format('legacy')
    assert pyvista.PICKLE_FORMAT == 'legacy'

    pyvista.set_pickle_format('xml')
    assert pyvista.PICKLE_FORMAT == 'xml'

    with pytest.raises(ValueError):
        pyvista.set_pickle_format('invalid_format')


def test_linkcode_resolve():
    assert linkcode_resolve('not-py', {}) is None
    link = linkcode_resolve('py', {'module': 'pyvista', 'fullname': 'pyvista.core.DataObject'})
    assert 'dataobject.py' in link
    assert '#L' in link

    # badmodule name
    assert linkcode_resolve('py', {'module': 'doesnotexist', 'fullname': 'foo.bar'}) is None

    assert (
        linkcode_resolve('py', {'module': 'pyvista', 'fullname': 'pyvista.not.an.object'}) is None
    )

    # test property
    link = linkcode_resolve('py', {'module': 'pyvista', 'fullname': 'pyvista.core.DataSet.points'})
    assert 'dataset.py' in link

    link = linkcode_resolve('py', {'module': 'pyvista', 'fullname': 'pyvista.core'})
    assert link.endswith('__init__.py')
