"""Test pyvista core utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import shutil
from unittest import mock
import warnings

import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import examples as ex
from pyvista.core.utilities import cells
from pyvista.core.utilities import fileio
from pyvista.core.utilities import fit_plane_to_points
from pyvista.core.utilities import principal_axes
from pyvista.core.utilities import transformations
from pyvista.core.utilities.arrays import _coerce_pointslike_arg
from pyvista.core.utilities.arrays import _coerce_transformlike_arg
from pyvista.core.utilities.arrays import _SerializedDictArray
from pyvista.core.utilities.arrays import copy_vtk_array
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import has_duplicates
from pyvista.core.utilities.arrays import raise_has_duplicates
from pyvista.core.utilities.arrays import vtk_id_list_to_array
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from pyvista.core.utilities.docs import linkcode_resolve
from pyvista.core.utilities.fileio import get_ext
from pyvista.core.utilities.helpers import is_inside_bounds
from pyvista.core.utilities.misc import assert_empty_kwargs
from pyvista.core.utilities.misc import check_valid_vector
from pyvista.core.utilities.misc import has_module
from pyvista.core.utilities.misc import no_new_attr
from pyvista.core.utilities.observers import Observer
from pyvista.core.utilities.points import vector_poly_data


def test_version():
    assert "major" in str(pv.vtk_version_info)
    ver = vtk.vtkVersion()
    assert ver.GetVTKMajorVersion() == pv.vtk_version_info.major
    assert ver.GetVTKMinorVersion() == pv.vtk_version_info.minor
    assert ver.GetVTKBuildVersion() == pv.vtk_version_info.micro
    ver_tup = (
        ver.GetVTKMajorVersion(),
        ver.GetVTKMinorVersion(),
        ver.GetVTKBuildVersion(),
    )
    assert ver_tup == pv.vtk_version_info
    assert pv.vtk_version_info >= (0, 0, 0)


def test_createvectorpolydata_error():
    orig = np.random.default_rng().random((3, 1))
    vec = np.random.default_rng().random((3, 1))
    with pytest.raises(ValueError):  # noqa: PT011
        vector_poly_data(orig, vec)


def test_createvectorpolydata_1D():
    orig = np.random.default_rng().random(3)
    vec = np.random.default_rng().random(3)
    vdata = vector_poly_data(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.point_data['vectors'])


def test_createvectorpolydata():
    orig = np.random.default_rng().random((100, 3))
    vec = np.random.default_rng().random((100, 3))
    vdata = vector_poly_data(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.point_data['vectors'])


@pytest.mark.parametrize(
    ('path', 'target_ext'),
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
        fnames = [Path(fname) for fname in fnames]
    types = (
        pv.PolyData,
        pv.PolyData,
        pv.UnstructuredGrid,
        pv.PolyData,
        pv.ImageData,
        pv.RectilinearGrid,
    )
    for i, filename in enumerate(fnames):
        obj = fileio.read(filename)
        assert isinstance(obj, types[i])
    # this is also tested for each mesh types init from file tests
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.npy'))
    arr = np.random.default_rng().random((10, 10))
    np.save(filename, arr)
    with pytest.raises(IOError):  # noqa: PT011
        _ = pv.read(filename)
    # read non existing file
    with pytest.raises(IOError):  # noqa: PT011
        _ = pv.read('this_file_totally_does_not_exist.vtk')
    # Now test reading lists of files as multi blocks
    multi = pv.read(fnames)
    assert isinstance(multi, pv.MultiBlock)
    assert multi.n_blocks == len(fnames)
    nested = [ex.planefile, [ex.hexbeamfile, ex.uniformfile]]

    multi = pv.read(nested)
    assert isinstance(multi, pv.MultiBlock)
    assert multi.n_blocks == 2
    assert isinstance(multi[1], pv.MultiBlock)
    assert multi[1].n_blocks == 2


def test_read_force_ext(tmpdir):
    fnames = (ex.antfile, ex.planefile, ex.hexbeamfile, ex.spherefile, ex.uniformfile, ex.rectfile)
    types = (
        pv.PolyData,
        pv.PolyData,
        pv.UnstructuredGrid,
        pv.PolyData,
        pv.ImageData,
        pv.RectilinearGrid,
    )

    dummy_extension = '.dummy'
    for fname, type_ in zip(fnames, types):
        path = Path(fname)
        root = str(path.parent / path.stem)
        original_ext = path.suffix
        _, name = os.path.split(root)
        new_fname = tmpdir / name + '.' + dummy_extension
        shutil.copy(fname, new_fname)
        data = fileio.read(new_fname, force_ext=original_ext)
        assert isinstance(data, type_)


@mock.patch('pyvista.BaseReader.read')
@mock.patch('pyvista.BaseReader.reader')
@mock.patch('pyvista.BaseReader.show_progress')
def test_read_progress_bar(mock_show_progress, mock_reader, mock_read):
    """Test passing attrs in read."""
    pv.read(ex.antfile, progress_bar=True)
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
    with pytest.raises(IOError):  # noqa: PT011
        fileio.read(fname, force_ext='.not_supported')


@mock.patch('pyvista.core.utilities.fileio.read_exodus')
def test_pyvista_read_exodus(read_exodus_mock):
    # check that reading a file with extension .e calls `read_exodus`
    # use the globefile as a dummy because pv.read() checks for the existence of the file
    pv.read(ex.globefile, force_ext='.e')
    args, kwargs = read_exodus_mock.call_args
    filename = args[0]
    assert filename == Path(ex.globefile)


def test_get_array_cell(hexbeam):
    carr = np.random.default_rng().random(hexbeam.n_cells)
    hexbeam.cell_data.set_array(carr, 'test_data')

    data = get_array(hexbeam, 'test_data', preference='cell')
    assert np.allclose(carr, data)


def test_get_array_point(hexbeam):
    parr = np.random.default_rng().random(hexbeam.n_points)
    hexbeam.point_data.set_array(parr, 'test_data')

    data = get_array(hexbeam, 'test_data', preference='point')
    assert np.allclose(parr, data)

    oarr = np.random.default_rng().random(hexbeam.n_points)
    hexbeam.point_data.set_array(oarr, 'other')

    data = get_array(hexbeam, 'other')
    assert np.allclose(oarr, data)


def test_get_array_field(hexbeam):
    hexbeam.clear_data()
    # no preference
    farr = np.random.default_rng().random(hexbeam.n_points * hexbeam.n_cells)
    hexbeam.field_data.set_array(farr, 'data')
    data = get_array(hexbeam, 'data')
    assert np.allclose(farr, data)

    # preference and multiple data
    hexbeam.point_data.set_array(np.random.default_rng().random(hexbeam.n_points), 'data')

    data = get_array(hexbeam, 'data', preference='field')
    assert np.allclose(farr, data)


def test_get_array_error(hexbeam):
    parr = np.random.default_rng().random(hexbeam.n_points)
    hexbeam.point_data.set_array(parr, 'test_data')

    # invalid inputs
    with pytest.raises(TypeError):
        get_array(hexbeam, 'test_data', preference={'invalid'})
    with pytest.raises(ValueError):  # noqa: PT011
        get_array(hexbeam, 'test_data', preference='invalid')
    with pytest.raises(ValueError, match='`preference` must be'):
        get_array(hexbeam, 'test_data', preference='row')


def test_get_array_none(hexbeam):
    arr = get_array(hexbeam, 'foo')
    assert arr is None


def get_array_vtk(hexbeam):
    # test raw VTK input
    grid_vtk = vtk.vtkUnstructuredGrid()
    grid_vtk.DeepCopy(hexbeam)
    get_array(grid_vtk, 'test_data')
    get_array(grid_vtk, 'foo')


def test_is_inside_bounds():
    data = ex.load_uniform()
    bnds = data.bounds
    assert is_inside_bounds((0.5, 0.5, 0.5), bnds)
    assert not is_inside_bounds((12, 5, 5), bnds)
    assert not is_inside_bounds((5, 12, 5), bnds)
    assert not is_inside_bounds((5, 5, 12), bnds)
    assert not is_inside_bounds((12, 12, 12), bnds)


def test_voxelize(uniform):
    vox = pv.voxelize(uniform, 0.5)
    assert vox.n_cells


def test_voxelize_non_uniform_density(uniform):
    vox = pv.voxelize(uniform, [0.5, 0.3, 0.2])
    assert vox.n_cells
    vox = pv.voxelize(uniform, np.array([0.5, 0.3, 0.2]))
    assert vox.n_cells


def test_voxelize_invalid_density(rectilinear):
    # test error when density is not length-3
    with pytest.raises(ValueError, match='not enough values to unpack'):
        pv.voxelize(rectilinear, [0.5, 0.3])
    # test error when density is not an array-like
    with pytest.raises(TypeError, match='expected number or array-like'):
        pv.voxelize(rectilinear, {0.5, 0.3})


def test_voxelize_throws_point_cloud(hexbeam):
    mesh = pv.PolyData(hexbeam.points)
    with pytest.raises(ValueError, match='must have faces'):
        pv.voxelize(mesh)


def test_voxelize_volume_default_density(uniform):
    expected = pv.voxelize_volume(uniform, density=uniform.length / 100).n_cells
    actual = pv.voxelize_volume(uniform).n_cells
    assert actual == expected


def test_voxelize_volume_invalid_density(rectilinear):
    with pytest.raises(TypeError, match='expected number or array-like'):
        pv.voxelize_volume(rectilinear, {0.5, 0.3})


def test_voxelize_volume_no_face_mesh(rectilinear):
    with pytest.raises(ValueError, match='must have faces'):
        pv.voxelize_volume(pv.PolyData())


def test_report():
    report = pv.Report(gpu=True)
    assert report is not None
    assert "GPU Details : None" not in report.__repr__()
    report = pv.Report(gpu=False)
    assert report is not None
    assert "GPU Details : None" in report.__repr__()


def test_line_segments_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    poly = pv.line_segments_from_points(points)
    assert poly.n_cells == 2
    assert poly.n_points == 4
    cells = poly.lines
    assert np.allclose(cells[:3], [2, 0, 1])
    assert np.allclose(cells[3:], [2, 2, 3])


def test_lines_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    poly = pv.lines_from_points(points)
    assert poly.n_cells == 2
    assert poly.n_points == 3
    cells = poly.lines
    assert np.allclose(cells[:3], [2, 0, 1])
    assert np.allclose(cells[3:], [2, 1, 2])


def test_grid_from_sph_coords():
    x = np.arange(0.0, 360.0, 40.0)  # longitude
    y = np.arange(0.0, 181.0, 60.0)  # colatitude
    z = [1]  # elevation (radius)
    g = pv.grid_from_sph_coords(x, y, z)
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
    g = pv.grid_from_sph_coords(x, y, z)
    assert g.n_cells == 48
    assert g.n_points == 108
    assert np.allclose(g.points[0], [0.0, 0.0, 10.0])


def test_transform_vectors_sph_to_cart():
    lon = np.arange(0.0, 360.0, 40.0)  # longitude
    lat = np.arange(0.0, 181.0, 60.0)  # colatitude
    lev = [1]  # elevation (radius)
    u, v = np.meshgrid(lon, lat, indexing="ij")
    w = u**2 - v**2
    uu, vv, ww = pv.transform_vectors_sph_to_cart(lon, lat, lev, u, v, w)
    assert np.allclose(
        [uu[-1, -1], vv[-1, -1], ww[-1, -1]],
        [67.80403533828323, 360.8359915416445, -70000.0],
    )


def test_vtkmatrix_to_from_array():
    rng = np.random.default_rng()
    array3x3 = rng.integers(0, 10, size=(3, 3))
    matrix = pv.vtkmatrix_from_array(array3x3)
    assert isinstance(matrix, vtk.vtkMatrix3x3)
    for i in range(3):
        for j in range(3):
            assert matrix.GetElement(i, j) == array3x3[i, j]

    array = pv.array_from_vtkmatrix(matrix)
    assert isinstance(array, np.ndarray)
    assert array.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            assert array[i, j] == matrix.GetElement(i, j)

    array4x4 = rng.integers(0, 10, size=(4, 4))
    matrix = pv.vtkmatrix_from_array(array4x4)
    assert isinstance(matrix, vtk.vtkMatrix4x4)
    for i in range(4):
        for j in range(4):
            assert matrix.GetElement(i, j) == array4x4[i, j]

    array = pv.array_from_vtkmatrix(matrix)
    assert isinstance(array, np.ndarray)
    assert array.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            assert array[i, j] == matrix.GetElement(i, j)

    # invalid cases
    with pytest.raises(ValueError):  # noqa: PT011
        matrix = pv.vtkmatrix_from_array(np.arange(3 * 4).reshape(3, 4))
    invalid = vtk.vtkTransform()
    with pytest.raises(TypeError):
        array = pv.array_from_vtkmatrix(invalid)


def test_assert_empty_kwargs():
    kwargs = {}
    assert assert_empty_kwargs(**kwargs)
    kwargs = {"foo": 6}
    with pytest.raises(TypeError):
        assert_empty_kwargs(**kwargs)
    kwargs = {"foo": 6, "goo": "bad"}
    with pytest.raises(TypeError):
        assert_empty_kwargs(**kwargs)


def test_convert_id_list():
    ids = np.array([4, 5, 8])
    id_list = vtk.vtkIdList()
    id_list.SetNumberOfIds(len(ids))
    for i, v in enumerate(ids):
        id_list.SetId(i, v)
    converted = vtk_id_list_to_array(id_list)
    assert np.allclose(converted, ids)


def test_progress_monitor():
    mesh = pv.Sphere()
    ugrid = mesh.delaunay_3d(progress_bar=True)
    assert isinstance(ugrid, pv.UnstructuredGrid)


def test_observer():
    msg = "KIND: In PATH, line 0\nfoo (ADDRESS): ALERT"
    obs = Observer()
    ret = obs.parse_message("foo")
    assert ret[3] == "foo"
    ret = obs.parse_message(msg)
    assert ret[3] == "ALERT"
    for kind in ["WARNING", "ERROR"]:
        obs.log_message(kind, "foo")
    # Pass positionally as that's what VTK will do
    obs(None, None, msg)
    assert obs.has_event_occurred()
    assert obs.get_message() == "ALERT"
    assert obs.get_message(etc=True) == msg

    alg = vtk.vtkSphereSource()
    alg.GetExecutive()
    obs.observe(alg)
    with pytest.raises(RuntimeError, match="algorithm"):
        obs.observe(alg)


def test_check_valid_vector():
    with pytest.raises(ValueError, match="length three"):
        check_valid_vector([0, 1])
    check_valid_vector([0, 1, 2])


def test_cells_dict_utils():
    # No pyvista object
    with pytest.raises(ValueError):  # noqa: PT011
        cells.get_mixed_cells(None)

    with pytest.raises(ValueError):  # noqa: PT011
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
    mesh = pv.StructuredGrid(x, y, z)
    x2, y2, z2 = np.meshgrid(np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5))
    mesh2 = pv.StructuredGrid(x2, y2, z2)

    alg = vtk.vtkStreamTracer()
    obs = pv.Observer()
    obs.observe(alg)
    alg.SetInputDataObject(mesh)
    alg.SetSourceData(mesh2)
    alg.Update()


def test_vtk_error_catcher():
    # raise_errors: False
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher()
    with error_catcher:
        _generate_vtk_err()
        _generate_vtk_err()
    assert len(error_catcher.events) == 2

    # raise_errors: False, no error
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher()
    with error_catcher:
        pass

    # raise_errors: True
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher(raise_errors=True)
    with pytest.raises(RuntimeError):
        with error_catcher:
            _generate_vtk_err()
    assert len(error_catcher.events) == 1

    # raise_errors: True, no error
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher(raise_errors=True)
    with error_catcher:
        pass


def test_axis_angle_rotation():
    # rotate points around body diagonal
    points = np.eye(3)
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
    with pytest.raises(ValueError):  # noqa: PT011
        transformations.axis_angle_rotation([1, 0, 0, 0], angle)
    with pytest.raises(ValueError):  # noqa: PT011
        transformations.axis_angle_rotation(axis, angle, point=[1, 0, 0, 0])
    with pytest.raises(ValueError):  # noqa: PT011
        transformations.axis_angle_rotation([0, 0, 0], angle)


@pytest.mark.parametrize(
    ("axis", "angle", "times"),
    [
        ([1, 0, 0], 90, 4),
        ([1, 0, 0], 180, 2),
        ([1, 0, 0], 270, 4),
        ([0, 1, 0], 90, 4),
        ([0, 1, 0], 180, 2),
        ([0, 1, 0], 270, 4),
        ([0, 0, 1], 90, 4),
        ([0, 0, 1], 180, 2),
        ([0, 0, 1], 270, 4),
    ],
)
def test_axis_angle_rotation_many_times(axis, angle, times):
    # yields the exact same input
    expect = np.eye(3)
    actual = expect.copy()
    trans = transformations.axis_angle_rotation(axis, angle)
    for _ in range(times):
        actual = transformations.apply_transformation_to_points(trans, actual)
    assert np.array_equal(actual, expect)


def test_reflection():
    # reflect points of a square across a diagonal
    points = np.array(
        [
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
        ],
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
    with pytest.raises(ValueError):  # noqa: PT011
        transformations.reflection([1, 0, 0, 0])
    with pytest.raises(ValueError):  # noqa: PT011
        transformations.reflection(normal, point=[1, 0, 0, 0])
    with pytest.raises(ValueError):  # noqa: PT011
        transformations.reflection([0, 0, 0])


def test_merge(sphere, cube, datasets):
    with pytest.raises(TypeError, match="Expected a sequence"):
        pv.merge(None)

    with pytest.raises(ValueError, match="Expected at least one"):
        pv.merge([])

    with pytest.raises(TypeError, match="Expected pyvista.DataSet"):
        pv.merge([None, sphere])

    # check polydata
    merged_poly = pv.merge([sphere, cube])
    assert isinstance(merged_poly, pv.PolyData)
    assert merged_poly.n_points == sphere.n_points + cube.n_points

    merged = pv.merge([sphere, sphere], merge_points=True)
    assert merged.n_points == sphere.n_points

    merged = pv.merge([sphere, sphere], merge_points=False)
    assert merged.n_points == sphere.n_points * 2

    # check unstructured
    merged_ugrid = pv.merge(datasets, merge_points=False)
    assert isinstance(merged_ugrid, pv.UnstructuredGrid)
    assert merged_ugrid.n_points == sum([ds.n_points for ds in datasets])
    # check main has priority
    sphere_a = sphere.copy()
    sphere_b = sphere.copy()
    sphere_a['data'] = np.zeros(sphere_a.n_points)
    sphere_b['data'] = np.ones(sphere_a.n_points)

    merged = pv.merge(
        [sphere_a, sphere_b],
        merge_points=True,
        main_has_priority=False,
    )
    assert np.allclose(merged['data'], 1)

    merged = pv.merge(
        [sphere_a, sphere_b],
        merge_points=True,
        main_has_priority=True,
    )
    assert np.allclose(merged['data'], 0)


def test_convert_array():
    arr = np.arange(4).astype('O')
    arr2 = pv.core.utilities.arrays.convert_array(arr, array_type=np.dtype('O'))
    assert arr2.GetNumberOfValues() == 4

    # https://github.com/pyvista/pyvista/issues/2370
    arr3 = pv.core.utilities.arrays.convert_array(
        pickle.loads(pickle.dumps(np.arange(4).astype('O'))),
        array_type=np.dtype('O'),
    )
    assert arr3.GetNumberOfValues() == 4

    # check lists work
    my_list = [1, 2, 3]
    arr4 = pv.core.utilities.arrays.convert_array(my_list)
    assert arr4.GetNumberOfValues() == len(my_list)

    # test string scalar is converted to string array with length on
    my_str = 'abc'
    arr5 = pv.core.utilities.arrays.convert_array(my_str)
    assert arr5.GetNumberOfValues() == 1
    arr6 = pv.core.utilities.arrays.convert_array(np.array(my_str))
    assert arr6.GetNumberOfValues() == 1


def test_has_duplicates():
    assert not has_duplicates(np.arange(100))
    assert has_duplicates(np.array([0, 1, 2, 2]))
    assert has_duplicates(np.array([[0, 1, 2], [0, 1, 2]]))

    with pytest.raises(ValueError):  # noqa: PT011
        raise_has_duplicates(np.array([0, 1, 2, 2]))


def test_copy_vtk_array():
    with pytest.raises(TypeError, match='Invalid type'):
        copy_vtk_array([1, 2, 3])

    value_0 = 10
    value_1 = 10
    arr = vtk.vtkFloatArray()
    arr.SetNumberOfValues(2)
    arr.SetValue(0, value_0)
    arr.SetValue(1, value_1)
    arr_copy = copy_vtk_array(arr, deep=True)
    assert arr_copy.GetNumberOfValues()
    assert value_0 == arr_copy.GetValue(0)

    arr_copy_shallow = copy_vtk_array(arr, deep=False)
    new_value = 5
    arr.SetValue(1, new_value)
    assert value_1 == arr_copy.GetValue(1)
    assert new_value == arr_copy_shallow.GetValue(1)


def test_cartesian_to_spherical():
    def polar2cart(r, phi, theta):
        return np.vstack(
            (r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)),
        ).T

    points = np.random.default_rng().random((1000, 3))
    x, y, z = points.T
    r, phi, theta = pv.cartesian_to_spherical(x, y, z)
    assert np.allclose(polar2cart(r, phi, theta), points)


def test_spherical_to_cartesian():
    points = np.random.default_rng().random((1000, 3))
    r, phi, theta = points.T
    x, y, z = pv.spherical_to_cartesian(r, phi, theta)
    assert np.allclose(pv.cartesian_to_spherical(x, y, z), points.T)


def test_set_pickle_format():
    pv.set_pickle_format('legacy')
    assert pv.PICKLE_FORMAT == 'legacy'

    pv.set_pickle_format('xml')
    assert pv.PICKLE_FORMAT == 'xml'

    with pytest.raises(ValueError):  # noqa: PT011
        pv.set_pickle_format('invalid_format')


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


def test_coerce_point_like_arg():
    # Test with Sequence
    point = [1.0, 2.0, 3.0]
    coerced_arg, singular = _coerce_pointslike_arg(point)
    assert isinstance(coerced_arg, np.ndarray)
    assert coerced_arg.shape == (1, 3)
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))
    assert singular

    # Test with 1D np.ndarray
    point = np.array([1.0, 2.0, 3.0])
    coerced_arg, singular = _coerce_pointslike_arg(point)
    assert isinstance(coerced_arg, np.ndarray)
    assert coerced_arg.shape == (1, 3)
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))
    assert singular

    # Test with (1, 3) np.ndarray
    point = np.array([[1.0, 2.0, 3.0]])
    coerced_arg, singular = _coerce_pointslike_arg(point)
    assert isinstance(coerced_arg, np.ndarray)
    assert coerced_arg.shape == (1, 3)
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))
    assert not singular

    # Test with 2D ndarray
    point = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coerced_arg, singular = _coerce_pointslike_arg(point)
    assert isinstance(coerced_arg, np.ndarray)
    assert coerced_arg.shape == (2, 3)
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert not singular


def test_coerce_point_like_arg_copy():
    # Sequence is always copied
    point = [1.0, 2.0, 3.0]
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=True)
    point[0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))

    point = [1.0, 2.0, 3.0]
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=False)
    point[0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))

    # 1D np.ndarray can be copied or not
    point = np.array([1.0, 2.0, 3.0])
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=True)
    point[0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))

    point = np.array([1.0, 2.0, 3.0])
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=False)
    point[0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[10.0, 2.0, 3.0]]))

    # 2D np.ndarray can be copied or not
    point = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=True)
    point[0, 0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    point = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=False)
    point[0, 0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[10.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_coerce_point_like_arg_errors():
    # wrong length sequence
    with pytest.raises(ValueError):  # noqa: PT011
        _coerce_pointslike_arg([1, 2])

    # wrong type
    with pytest.raises(TypeError):
        # allow Sequence but not Iterable
        _coerce_pointslike_arg({1, 2, 3})

    # wrong length ndarray
    with pytest.raises(ValueError):  # noqa: PT011
        _coerce_pointslike_arg(np.empty(4))
    with pytest.raises(ValueError):  # noqa: PT011
        _coerce_pointslike_arg(np.empty([2, 4]))

    # wrong ndim ndarray
    with pytest.raises(ValueError):  # noqa: PT011
        _coerce_pointslike_arg(np.empty([1, 3, 3]))


def test_coerce_points_like_args_does_not_copy():
    source = np.random.default_rng().random((100, 3))
    output, _ = _coerce_pointslike_arg(source)  # test that copy=False is default
    output /= 2
    assert np.array_equal(output, source)
    assert np.may_share_memory(output, source)


def test_has_module():
    assert has_module('pytest')
    assert not has_module('not_a_module')


def test_fit_plane_to_points():
    points = ex.load_airplane().points
    plane, center, normal = fit_plane_to_points(points, return_meta=True)

    assert np.allclose(normal, [-2.5999512e-08, 0.121780515, -0.99255705])
    assert np.allclose(center, [896.9954860028446, 686.6470205328502, 78.13187948615939])
    assert np.allclose(
        plane.bounds,
        [
            139.06036376953125,
            1654.9306640625,
            38.0776252746582,
            1335.2164306640625,
            -1.4434913396835327,
            157.70724487304688,
        ],
    )


DEFAULT_PRINCIPAL_AXES = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]

CASE_RANK0 = (
    0,
    [[0, 0, 0], [0, 0, 0]],
    [DEFAULT_PRINCIPAL_AXES],
)
CASE_RANK1 = (
    1,
    [[0, 0, 0], [1, 1, 1]],
    [
        [0.57735027, 0.57735027, 0.57735027],
        [0.11435482, 0.64295988, -0.75731471],
        [-0.80844891, 0.50325864, 0.30519027],
    ],
)

CASE_RANK2 = (
    2,
    [[0, 1, 3], [2, 0, 0]],
    [
        [0.53452248, -0.26726124, -0.80178373],
        [0.79735937, -0.15503524, 0.58325133],
        [-0.28018521, -0.95107071, 0.13023343],
    ],
)
CASE_RANK3 = (
    3,
    pv.examples.load_airplane().points,
    [
        [8.1159601e-07, -9.9255711e-01, -1.2178054e-01],
        [-1.0000000e00, -8.0911440e-07, -6.9828459e-08],
        [-2.9225653e-08, 1.2178054e-01, -9.9255711e-01],
    ],
)


@pytest.mark.parametrize(
    ('rank', 'points', 'expected_axes'),
    [CASE_RANK0, CASE_RANK1, CASE_RANK2, CASE_RANK3],
    ids=['rank0', 'rank1', 'rank2', 'rank3'],
)
def test_principal_axes(rank, points, expected_axes):
    assert np.linalg.matrix_rank(points) == rank

    axes = principal_axes(points)
    assert np.allclose(axes, expected_axes)
    assert np.array_equal(np.cross(axes[0], axes[1]), axes[2])
    assert np.allclose(np.linalg.norm(axes[0]), 1)
    assert np.allclose(np.linalg.norm(axes[1]), 1)
    assert np.allclose(np.linalg.norm(axes[2]), 1)

    assert type(axes) is np.ndarray


def test_principal_axes_empty():
    axes = principal_axes(np.empty((0, 3)))
    assert np.allclose(axes, DEFAULT_PRINCIPAL_AXES)


def test_principal_axes_single_point():
    axes = principal_axes([1, 2, 3])
    assert np.allclose(axes, DEFAULT_PRINCIPAL_AXES)


def test_principal_axes_vectors_many_points():
    n_points = 2_000_000
    points = np.random.default_rng().random((n_points, 3))
    axes = pv.principal_axes(points)
    assert np.any(axes)
    assert np.all(axes != 0)


@pytest.mark.parametrize(
    'transform_like',
    [
        np.array(np.eye(3)),
        np.array(np.eye(4)),
        vtkmatrix_from_array(np.eye(3)),
        vtkmatrix_from_array(np.eye(4)),
        vtk.vtkTransform(),
    ],
)
def test_coerce_transformlike_arg(transform_like):
    result = _coerce_transformlike_arg(transform_like)
    assert np.array_equal(result, np.eye(4))


def test_coerce_transformlike_arg_raises():
    with pytest.raises(ValueError, match="must be 3x3 or 4x4"):
        _coerce_transformlike_arg(np.array([1, 2, 3]))
    with pytest.raises(TypeError, match="must be one of"):
        _coerce_transformlike_arg([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(TypeError, match="must be one of"):
        _coerce_transformlike_arg("abc")


@pytest.fixture()
def no_new_attr_subclass():
    @no_new_attr
    class A: ...

    class B(A):
        _new_attr_exceptions = 'eggs'

        def __init__(self):
            self.eggs = 'ham'

    return B


def test_no_new_attr_subclass(no_new_attr_subclass):
    obj = no_new_attr_subclass()
    assert obj
    msg = 'Attribute "_eggs" does not exist and cannot be added to type B'
    with pytest.raises(AttributeError, match=msg):
        obj._eggs = 'ham'


@pytest.fixture()
def serial_dict_empty():
    return _SerializedDictArray()


@pytest.fixture()
def serial_dict_with_foobar():
    serial_dict = _SerializedDictArray()
    serial_dict.data = dict(foo='bar')
    return serial_dict


def test_serial_dict_init():
    # empty init
    serial_dict = _SerializedDictArray()
    assert serial_dict == {}
    assert repr(serial_dict) == '{}'

    # init from dict
    new_dict = dict(ham='eggs')
    serial_dict = _SerializedDictArray(new_dict)
    assert serial_dict['ham'] == 'eggs'
    assert repr(serial_dict) == '{"ham": "eggs"}'

    # init from UserDict
    serial_dict = _SerializedDictArray(serial_dict)
    assert serial_dict['ham'] == 'eggs'
    assert repr(serial_dict) == '{"ham": "eggs"}'

    # init from JSON string
    json_dict = json.dumps(new_dict)
    serial_dict = _SerializedDictArray(json_dict)
    assert serial_dict['ham'] == 'eggs'
    assert repr(serial_dict) == '{"ham": "eggs"}'


def test_serial_dict_as_dict(serial_dict_with_foobar):
    assert not isinstance(serial_dict_with_foobar, dict)
    actual_dict = dict(serial_dict_with_foobar)
    assert isinstance(actual_dict, dict)
    assert actual_dict == serial_dict_with_foobar.data


def test_serial_dict_overrides__setitem__(serial_dict_empty):
    serial_dict_empty['foo'] = 'bar'
    assert repr(serial_dict_empty) == '{"foo": "bar"}'


def test_serial_dict_overrides__delitem__(serial_dict_with_foobar):
    del serial_dict_with_foobar['foo']
    assert repr(serial_dict_with_foobar) == '{}'


def test_serial_dict_overrides__setattr__(serial_dict_empty):
    serial_dict_empty.data = dict(foo='bar')
    assert repr(serial_dict_empty) == '{"foo": "bar"}'


def test_serial_dict_overrides_popitem(serial_dict_with_foobar):
    serial_dict_with_foobar['ham'] = 'eggs'
    item = serial_dict_with_foobar.popitem()
    assert item == ('foo', 'bar')
    assert repr(serial_dict_with_foobar) == '{"ham": "eggs"}'


def test_serial_dict_overrides_pop(serial_dict_with_foobar):
    item = serial_dict_with_foobar.pop('foo')
    assert item == 'bar'
    assert repr(serial_dict_with_foobar) == '{}'


def test_serial_dict_overrides_update(serial_dict_empty):
    serial_dict_empty.update(dict(foo='bar'))
    assert repr(serial_dict_empty) == '{"foo": "bar"}'


def test_serial_dict_overrides_clear(serial_dict_with_foobar):
    serial_dict_with_foobar.clear()
    assert repr(serial_dict_with_foobar) == '{}'


def test_serial_dict_overrides_setdefault(serial_dict_empty, serial_dict_with_foobar):
    serial_dict_empty.setdefault('foo', 42)
    assert repr(serial_dict_empty) == '{"foo": 42}'
    serial_dict_with_foobar.setdefault('foo', 42)
    assert repr(serial_dict_with_foobar) == '{"foo": "bar"}'
