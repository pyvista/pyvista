"""Test pyvista core utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import platform
import re
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
from pyvista.core.utilities import fit_line_to_points
from pyvista.core.utilities import fit_plane_to_points
from pyvista.core.utilities import principal_axes
from pyvista.core.utilities import transformations
from pyvista.core.utilities.arrays import _coerce_pointslike_arg
from pyvista.core.utilities.arrays import _SerializedDictArray
from pyvista.core.utilities.arrays import copy_vtk_array
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import has_duplicates
from pyvista.core.utilities.arrays import raise_has_duplicates
from pyvista.core.utilities.arrays import vtk_id_list_to_array
from pyvista.core.utilities.docs import linkcode_resolve
from pyvista.core.utilities.fileio import get_ext
from pyvista.core.utilities.helpers import is_inside_bounds
from pyvista.core.utilities.misc import assert_empty_kwargs
from pyvista.core.utilities.misc import check_valid_vector
from pyvista.core.utilities.misc import has_module
from pyvista.core.utilities.misc import no_new_attr
from pyvista.core.utilities.observers import Observer
from pyvista.core.utilities.points import vector_poly_data
from pyvista.core.utilities.transform import Transform
from pyvista.plotting.prop3d import _orientation_as_rotation_matrix
from tests.conftest import NUMPY_VERSION_INFO


@pytest.fixture
def transform():
    return Transform()


def test_version():
    assert 'major' in str(pv.vtk_version_info)
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
        ('/data/mesh.stl', '.stl'),
        ('/data/image.nii.gz', '.nii.gz'),
        ('/data/other.gz', '.gz'),
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
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.npy'))
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
        warnings.simplefilter('ignore')
        data = fileio.read(fname, force_ext='.vts')
    assert data.n_points == 0

    # try to read a .ply file as .vtm
    # vtkXMLMultiBlockDataReader throws a VTK error about the validity of the XML file
    # the returned dataset is empty
    fname = ex.planefile
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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


@pytest.mark.parametrize('function', [pv.voxelize_volume, pv.voxelize])
def test_voxelize_enclosed_bounds(function, ant):
    vox = function(ant, density=0.9, enclosed=True)

    assert vox.bounds.x_min <= ant.bounds.x_min
    assert vox.bounds.y_min <= ant.bounds.y_min
    assert vox.bounds.z_min <= ant.bounds.z_min

    assert vox.bounds.x_max >= ant.bounds.x_max
    assert vox.bounds.y_max >= ant.bounds.y_max
    assert vox.bounds.z_max >= ant.bounds.z_max


@pytest.mark.parametrize('function', [pv.voxelize_volume, pv.voxelize])
def test_voxelize_fit_bounds(function, uniform):
    vox = function(uniform, density=0.9, fit_bounds=True)

    assert np.isclose(vox.bounds.x_min, uniform.bounds.x_min)
    assert np.isclose(vox.bounds.y_min, uniform.bounds.y_min)
    assert np.isclose(vox.bounds.z_min, uniform.bounds.z_min)

    assert np.isclose(vox.bounds.x_max, uniform.bounds.x_max)
    assert np.isclose(vox.bounds.y_max, uniform.bounds.y_max)
    assert np.isclose(vox.bounds.z_max, uniform.bounds.z_max)


def test_report():
    report = pv.Report(gpu=True)
    assert report is not None
    assert 'GPU Details : None' not in report.__repr__()
    report = pv.Report(gpu=False)
    assert report is not None
    assert 'GPU Details : None' in report.__repr__()


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
    u, v = np.meshgrid(lon, lat, indexing='ij')
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
    kwargs = {'foo': 6}
    with pytest.raises(TypeError):
        assert_empty_kwargs(**kwargs)
    kwargs = {'foo': 6, 'goo': 'bad'}
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
    msg = 'KIND: In PATH, line 0\nfoo (ADDRESS): ALERT'
    obs = Observer()
    ret = obs.parse_message('foo')
    assert ret[3] == 'foo'
    ret = obs.parse_message(msg)
    assert ret[3] == 'ALERT'
    for kind in ['WARNING', 'ERROR']:
        obs.log_message(kind, 'foo')
    # Pass positionally as that's what VTK will do
    obs(None, None, msg)
    assert obs.has_event_occurred()
    assert obs.get_message() == 'ALERT'
    assert obs.get_message(etc=True) == msg

    alg = vtk.vtkSphereSource()
    alg.GetExecutive()
    obs.observe(alg)
    with pytest.raises(RuntimeError, match='algorithm'):
        obs.observe(alg)


def test_check_valid_vector():
    with pytest.raises(ValueError, match='length three'):
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
    ('axis', 'angle', 'times'),
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
    with pytest.raises(TypeError, match='Expected a sequence'):
        pv.merge(None)

    with pytest.raises(ValueError, match='Expected at least one'):
        pv.merge([])

    with pytest.raises(TypeError, match='Expected pyvista.DataSet'):
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


def test_fit_plane_to_points_resolution(airplane):
    DEFAULT_RESOLUTION = 10
    plane = fit_plane_to_points(airplane.points)
    assert plane.n_points == (DEFAULT_RESOLUTION + 1) ** 2

    resolution = (1.0, 2.0)  # Test with integer-valued floats
    plane = fit_plane_to_points(airplane.points, resolution=resolution)
    assert plane.n_points == (resolution[0] + 1) * (resolution[1] + 1)


def test_fit_plane_to_points():
    # Fit a plane to a plane's points
    center = (1, 2, 3)
    direction = np.array((4.0, 5.0, 6.0))
    direction /= np.linalg.norm(direction)
    expected_plane = pv.Plane(center=center, direction=direction, i_size=2, j_size=3)
    fitted_plane, fitted_center, fitted_normal = fit_plane_to_points(
        expected_plane.points, return_meta=True
    )

    # Test bounds
    assert np.allclose(fitted_plane.bounds, expected_plane.bounds, atol=1e-6)

    # Test center
    assert np.allclose(fitted_plane.center, center)
    assert np.allclose(fitted_center, center)
    assert np.allclose(fitted_plane.points.mean(axis=0), center)

    # Test normal
    assert np.allclose(fitted_normal, direction)
    assert np.allclose(fitted_plane.point_normals.mean(axis=0), direction)

    flipped_normal = direction * -1
    _, _, new_normal = fit_plane_to_points(
        expected_plane.points, return_meta=True, init_normal=flipped_normal
    )
    assert np.allclose(new_normal, flipped_normal)


def test_fit_line_to_points():
    # Fit a line to a line's points
    point_a = (1, 2, 3)
    point_b = (4, 5, 6)
    resolution = 42
    expected_line = pv.Line(point_a, point_b, resolution=resolution)
    fitted_line, length, direction = fit_line_to_points(
        expected_line.points, resolution=resolution, return_meta=True
    )

    assert np.allclose(fitted_line.bounds, expected_line.bounds)
    assert np.allclose(fitted_line.points[0], point_a)
    assert np.allclose(fitted_line.points[-1], point_b)
    assert np.allclose(direction, np.abs(pv.principal_axes(fitted_line.points)[0]))
    assert np.allclose(length, fitted_line.length)

    fitted_line = fit_line_to_points(expected_line.points, resolution=resolution, return_meta=False)
    assert np.allclose(fitted_line.bounds, expected_line.bounds)


# Default output from `np.linalg.eigh`
DEFAULT_PRINCIPAL_AXES = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]


CASE_0 = (  # coincidental points
    [[0, 0, 0], [0, 0, 0]],
    [DEFAULT_PRINCIPAL_AXES],
)
CASE_1 = (  # non-coincidental points
    [[0, 0, 0], [1, 0, 0]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
)

CASE_2 = (  # non-collinear points
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [
        [0.0, -0.70710678, 0.70710678],
        [-0.81649658, 0.40824829, 0.40824829],
        [-0.57735027, -0.57735027, -0.57735027],
    ],
)
CASE_3 = (  # non-coplanar points
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]],
    [
        [-0.57735027, -0.57735027, -0.57735027],
        [0.0, -0.70710678, 0.70710678],
        [-0.81649658, 0.40824829, 0.40824829],
    ],
)

is_arm_mac = platform.system() == 'Darwin' and platform.machine() == 'arm64'


@pytest.mark.skipif(
    NUMPY_VERSION_INFO < (1, 26) or is_arm_mac, reason='Different results for some tests.'
)
@pytest.mark.parametrize(
    ('points', 'expected_axes'),
    [CASE_0, CASE_1, CASE_2, CASE_3],
    ids=['case0', 'case1', 'case2', 'case3'],
)
def test_principal_axes(points, expected_axes):
    axes = principal_axes(points)
    assert np.allclose(axes, expected_axes, atol=1e-7)

    assert np.allclose(np.cross(axes[0], axes[1]), axes[2])
    assert np.allclose(np.linalg.norm(axes, axis=1), 1)
    assert isinstance(axes, np.ndarray)

    _, std = principal_axes(points, return_std=True)
    assert std[0] >= std[1]
    if not np.isnan(std[2]):
        assert std[1] >= std[2]
    assert isinstance(std, np.ndarray)


def test_principal_axes_return_std():
    # Create axis-aligned normally distributed points
    rng = np.random.default_rng(seed=42)
    n = 100_000
    std_in = np.array([3, 2, 1])
    normal_points = rng.normal(size=(n, 3)) * std_in

    _, std_out = principal_axes(normal_points, return_std=True)

    # Test output matches numpy std
    std_numpy = np.std(normal_points, axis=0)
    assert np.allclose(std_out, std_numpy, atol=1e-4)

    # Test output matches input std
    assert np.allclose(std_out, std_in, atol=0.02)

    # Test ratios of input sizes match ratios of output std
    ratios_in = std_in / sum(std_in)
    ratios_out = std_out / sum(std_out)
    assert np.allclose(ratios_in, ratios_out, atol=0.02)


def test_principal_axes_empty():
    axes = principal_axes(np.empty((0, 3)))
    assert np.allclose(axes, DEFAULT_PRINCIPAL_AXES)


def test_principal_axes_single_point():
    axes = principal_axes([1, 2, 3])
    assert np.allclose(axes, DEFAULT_PRINCIPAL_AXES)


@pytest.fixture
def one_million_points():
    return np.random.default_rng().random((1_000_000, 3))


def test_principal_axes_success_with_many_points(one_million_points):
    # Use many points to verify no memory errors are raised
    axes = pv.principal_axes(one_million_points)
    assert isinstance(axes, np.ndarray)


def test_fit_plane_to_points_success_with_many_points(one_million_points):
    # Use many points to verify no memory errors are raised
    plane = pv.fit_plane_to_points(one_million_points)
    assert isinstance(plane, pv.PolyData)


@pytest.fixture
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


@pytest.fixture
def serial_dict_empty():
    return _SerializedDictArray()


@pytest.fixture
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


SCALE = 2
ROTATION = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # rotate 90 deg about z axis
VECTOR = (1, 2, 3)
ANGLE = 30


@pytest.mark.parametrize('scale_args', [(SCALE,), (SCALE, SCALE, SCALE), [(SCALE, SCALE, SCALE)]])
def test_transform_scale(transform, scale_args):
    transform.scale(*scale_args)
    actual = transform.matrix
    expected = np.diag((SCALE, SCALE, SCALE, 1))
    assert np.array_equal(actual, expected)
    assert transform.n_transformations == 1

    identity = transform.matrix @ transform.inverse_matrix
    assert np.array_equal(identity, np.eye(4))


@pytest.mark.parametrize('translate_args', [np.array(VECTOR), np.array([VECTOR])])
def test_transform_translate(transform, translate_args):
    transform.translate(*translate_args)
    actual = transform.matrix
    expected = np.eye(4)
    expected[:3, 3] = VECTOR
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.array_equal(identity, np.eye(4))


@pytest.mark.parametrize('reflect_args', [VECTOR, [VECTOR]])
def test_transform_reflect(transform, reflect_args):
    transform.reflect(*reflect_args)
    actual = transform.matrix
    expected = transformations.reflection(VECTOR)
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


@pytest.mark.parametrize(
    ('method', 'vector'), [('flip_x', (1, 0, 0)), ('flip_y', (0, 1, 0)), ('flip_z', (0, 0, 1))]
)
def test_transform_flip_xyz(transform, method, vector):
    getattr(transform, method)()
    actual = transform.matrix
    expected = transformations.reflection(vector)
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_rotate(transform):
    transform.rotate(ROTATION)
    actual = transform.matrix
    expected = np.eye(4)
    expected[:3, :3] = ROTATION
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.array_equal(identity, np.eye(4))


@pytest.mark.parametrize('multiply_mode', ['post', 'pre'])
@pytest.mark.parametrize(
    ('method', 'args'),
    [
        ('scale', (SCALE,)),
        ('reflect', (VECTOR,)),
        ('flip_x', ()),
        ('flip_y', ()),
        ('flip_z', ()),
        ('rotate', (ROTATION,)),
        ('rotate_x', (ANGLE,)),
        ('rotate_y', (ANGLE,)),
        ('rotate_z', (ANGLE,)),
        ('rotate_vector', (VECTOR, ANGLE)),
    ],
)
def test_transform_with_point(transform, multiply_mode, method, args):
    func = getattr(Transform, method)
    vector = np.array(VECTOR)

    transform.multiply_mode = multiply_mode
    transform.point = vector
    func(transform, *args)

    expected_transform = Transform().translate(-vector)
    func(expected_transform, *args)
    expected_transform.translate(vector)

    assert np.array_equal(transform.matrix, expected_transform.matrix)
    assert transform.n_transformations == 3

    # Test override point with kwarg
    vector2 = vector * 2  # new point
    transform.identity()  # reset
    func(transform, *args, point=vector2)  # override point

    expected_transform = Transform().translate(-vector2)
    func(expected_transform, *args)
    expected_transform.translate(vector2)

    assert np.array_equal(transform.matrix, expected_transform.matrix)
    assert transform.n_transformations == 3


def test_transform_rotate_x(transform):
    transform.rotate_x(ANGLE)
    actual = transform.matrix
    expected = transformations.axis_angle_rotation((1, 0, 0), ANGLE)
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_rotate_y(transform):
    transform.rotate_y(ANGLE)
    actual = transform.matrix
    expected = transformations.axis_angle_rotation((0, 1, 0), ANGLE)
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_rotate_z(transform):
    transform.rotate_z(ANGLE)
    actual = transform.matrix
    expected = transformations.axis_angle_rotation((0, 0, 1), ANGLE)
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_rotate_vector(transform):
    transform.rotate_vector(VECTOR, ANGLE)
    actual = transform.matrix
    expected = transformations.axis_angle_rotation(VECTOR, ANGLE)
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_concatenate_vtkmatrix(transform):
    scale_array = np.diag((1, 2, 3, 1))
    vtkmatrix = pv.vtkmatrix_from_array(scale_array)
    transform.concatenate(vtkmatrix)
    actual = transform.matrix
    expected = scale_array
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.array_equal(identity, np.eye(4))


def test_transform_invert(transform):
    assert transform.is_inverted is False

    # Add a transformation and check its output
    transform.scale(SCALE)
    inverse = transform.inverse_matrix
    transform.invert()
    assert transform.is_inverted is True
    assert np.array_equal(inverse, transform.matrix)

    transform.invert()
    assert transform.is_inverted is False


@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize(
    ('obj', 'return_self', 'return_type', 'return_dtype'),
    [
        (list(VECTOR), False, np.ndarray, float),
        (VECTOR, False, np.ndarray, float),
        (np.array(VECTOR), False, np.ndarray, float),
        (np.array([VECTOR]), False, np.ndarray, float),
        (np.array(VECTOR, dtype=float), True, np.ndarray, float),
        (np.array([VECTOR], dtype=float), True, np.ndarray, float),
        (pv.PolyData(np.atleast_2d(VECTOR)), True, pv.PolyData, np.float32),
        (pv.PolyData(np.atleast_2d(VECTOR).astype(int)), True, pv.PolyData, np.float32),
        (pv.PolyData(np.atleast_2d(VECTOR).astype(float)), True, pv.PolyData, float),
        (
            pv.MultiBlock([pv.PolyData(np.atleast_2d(VECTOR).astype(float))]),
            True,
            pv.MultiBlock,
            float,
        ),
    ],
    ids=[
        'list-int',
        'tuple-int',
        'array1d-int',
        'array2d-int',
        'array1d-float',
        'array2d-float',
        'polydata-float32',
        'polydata-int',
        'polydata-float',
        'multiblock-float',
    ],
)
def test_transform_apply(transform, obj, return_self, return_type, return_dtype, copy):
    def _get_points_from_object(obj_):
        return (
            obj_.points
            if isinstance(obj_, pv.DataSet)
            else obj_[0].points
            if isinstance(obj_, pv.MultiBlock)
            else obj_
        )

    points_in_array = np.array(_get_points_from_object(obj))
    out = transform.scale(SCALE).apply(obj, copy=copy, transform_all_input_vectors=True)

    if not copy and return_self:
        assert out is obj
    else:
        assert out is not obj
    assert isinstance(out, return_type)

    points_out = _get_points_from_object(out)
    assert isinstance(points_out, np.ndarray)
    assert points_out.dtype == return_dtype
    assert np.array_equal(points_in_array * SCALE, points_out)

    inverted = transform.apply(out, inverse=True)
    inverted_points = _get_points_from_object(inverted)
    assert np.array_equal(inverted_points, points_in_array)
    assert not transform.is_inverted


@pytest.mark.parametrize('attr', ['matrix_list', 'inverse_matrix_list'])
def test_transform_matrix_list(transform, attr):
    matrix_list = getattr(transform, attr)
    assert isinstance(matrix_list, list)
    assert len(matrix_list) == 0
    assert transform.n_transformations == 0

    transform.scale(SCALE)
    matrix_list = getattr(transform, attr)
    assert len(matrix_list) == 1
    assert transform.n_transformations == 1
    assert isinstance(matrix_list[0], np.ndarray)
    assert matrix_list[0].shape == (4, 4)

    transform.rotate([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    matrix_list = getattr(transform, attr)
    assert len(matrix_list) == 2

    identity = transform.matrix_list[0] @ transform.inverse_matrix_list[0]
    assert np.array_equal(identity, np.eye(4))


@pytest.fixture
def transformed_actor():
    actor = pv.Actor()
    actor.position = (-0.5, -0.5, 1)
    actor.orientation = (10, 20, 30)
    actor.scale = (1.5, 2, 2.5)
    actor.origin = (2, 1.5, 1)
    actor.user_matrix = pv.array_from_vtkmatrix(actor.GetMatrix())
    return actor


@pytest.mark.parametrize('override_mode', ['pre', 'post'])
@pytest.mark.parametrize('object_mode', ['pre', 'post'])
def test_transform_multiply_mode_override(transform, transformed_actor, object_mode, override_mode):
    # This test validates multiply mode by performing the same transformations
    # applied by `Prop3D` objects and comparing the results
    transform.multiply_mode = object_mode

    # Center data at the origin
    transform.translate(np.array(transformed_actor.origin) * -1, multiply_mode=override_mode)

    # Scale and rotate
    transform.scale(transformed_actor.scale, multiply_mode=override_mode)
    rotation = _orientation_as_rotation_matrix(transformed_actor.orientation)
    transform.rotate(rotation, multiply_mode=override_mode)

    # Move to position
    transform.translate(np.array(transformed_actor.origin), multiply_mode=override_mode)
    transform.translate(transformed_actor.position, multiply_mode=override_mode)

    # Apply user matrix
    transform.concatenate(transformed_actor.user_matrix, multiply_mode=override_mode)

    # Check result
    transform_matrix = transform.matrix
    actor_matrix = pv.array_from_vtkmatrix(transformed_actor.GetMatrix())
    if override_mode == 'post':
        assert np.allclose(transform_matrix, actor_matrix)
    else:
        # Pre-multiplication produces a totally different result
        assert not np.allclose(transform_matrix, actor_matrix)


def test_transform_multiply_mode(transform):
    assert transform.multiply_mode == 'post'
    transform.multiply_mode = 'pre'
    assert transform.multiply_mode == 'pre'

    transform.post_multiply()
    assert transform.multiply_mode == 'post'
    transform.pre_multiply()
    assert transform.multiply_mode == 'pre'


def test_transform_identity(transform):
    transform.scale(2)
    assert not np.array_equal(transform.matrix, np.eye(4))
    transform.identity()
    assert np.array_equal(transform.matrix, np.eye(4))


def test_transform_init():
    matrix = np.diag((SCALE, SCALE, SCALE, 1))
    transform = Transform(matrix)
    assert np.array_equal(transform.matrix, matrix)


def test_transform_chain_methods():
    eye3 = np.eye(3)
    eye4 = np.eye(4)
    ones = (1, 1, 1)
    zeros = (0, 0, 0)
    matrix = (
        Transform()
        .reflect(ones)
        .flip_x()
        .flip_y()
        .flip_z()
        .rotate_x(0)
        .rotate_y(0)
        .rotate_z(0)
        .rotate_vector(ones, 0)
        .identity()
        .scale(ones)
        .translate(zeros)
        .rotate(eye3)
        .concatenate(eye4)
        .invert()
        .post_multiply()
        .pre_multiply()
        .matrix
    )
    assert np.array_equal(matrix, eye4)


def test_transform_add():
    scale = Transform().scale(SCALE)
    translate = Transform().translate(VECTOR)

    transform = pv.Transform().post_multiply().translate(VECTOR).scale(SCALE)
    transform_add = translate + scale
    assert np.array_equal(transform_add.matrix, transform.matrix)

    # Validate with numpy matmul
    matrix_numpy = scale.matrix @ translate.matrix
    assert np.array_equal(transform_add.matrix, matrix_numpy)


@pytest.mark.parametrize(
    'other', [VECTOR, Transform().translate(VECTOR), Transform().translate(VECTOR).matrix]
)
def test_transform_add_other(other):
    transform_base = pv.Transform().post_multiply().scale(SCALE)
    # Translate with `translate` and `+`
    transform_translate = transform_base.copy().translate(VECTOR)
    transform_add = transform_base + other
    assert np.array_equal(transform_add.matrix, transform_translate.matrix)

    # Test multiply mode override to ensure post-multiply is always used
    transform_add = transform_base.pre_multiply() + other
    assert np.array_equal(transform_add.matrix, transform_translate.matrix)


def test_transform_radd():
    transform_base = pv.Transform().pre_multiply().scale(SCALE)
    # Translate with `translate` and `+`
    transform_translate = transform_base.copy().translate(VECTOR)
    transform_add = VECTOR + transform_base
    assert np.array_equal(transform_add.matrix, transform_translate.matrix)

    # Test multiply mode override to ensure post-multiply is always used
    transform_add = VECTOR + transform_base.post_multiply()
    assert np.array_equal(transform_add.matrix, transform_translate.matrix)


@pytest.mark.parametrize('scale_factor', [SCALE, (SCALE, SCALE, SCALE)])
def test_transform_mul(scale_factor):
    transform_base = pv.Transform().post_multiply().translate(VECTOR)
    # Scale with `scale` and `*`
    transform_scale = transform_base.copy().scale(scale_factor)
    transform_mul = transform_base * scale_factor
    assert np.array_equal(transform_mul.matrix, transform_scale.matrix)

    # Test multiply mode override to ensure post-multiply is always used
    transform_mul = transform_base.pre_multiply() * scale_factor
    assert np.array_equal(transform_mul.matrix, transform_scale.matrix)


@pytest.mark.parametrize('scale_factor', [SCALE, (SCALE, SCALE, SCALE)])
def test_transform_rmul(scale_factor):
    transform_base = pv.Transform().pre_multiply().translate(VECTOR)
    # Scale with `scale` and `*`
    transform_scale = transform_base.copy().scale(scale_factor)
    transform_mul = scale_factor * transform_base
    assert np.array_equal(transform_mul.matrix, transform_scale.matrix)

    # Test multiply mode override to ensure pre-multiply is always used
    transform_scale = transform_base.copy().scale(scale_factor)
    transform_mul = scale_factor * transform_base.post_multiply()
    assert np.array_equal(transform_mul.matrix, transform_scale.matrix)


def test_transform_matmul():
    scale = Transform().scale(SCALE)
    translate = Transform().translate(VECTOR)

    transform = pv.Transform().pre_multiply().translate(VECTOR).scale(SCALE)
    transform_matmul = translate @ scale
    assert np.array_equal(transform_matmul.matrix, transform.matrix)

    # Test multiply mode override to ensure pre-multiply is always used
    transform_matmul = translate.post_multiply() @ scale.post_multiply()
    assert np.array_equal(transform_matmul.matrix, transform.matrix)

    # Validate with numpy matmul
    matrix_numpy = translate.matrix @ scale.matrix
    assert np.array_equal(transform_matmul.matrix, matrix_numpy)


def test_transform_add_raises():
    match = (
        "Unsupported operand value(s) for +: 'Transform' and 'int'\n"
        'The right-side argument must be a length-3 vector or have 3x3 or 4x4 shape.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.Transform() + 1

    match = (
        "Unsupported operand type(s) for +: 'Transform' and 'dict'\n"
        'The right-side argument must be transform-like.'
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        pv.Transform() + {}


def test_transform_radd_raises():
    match = (
        "Unsupported operand value(s) for +: 'int' and 'Transform'\n"
        'The left-side argument must be a length-3 vector.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        1 + pv.Transform()

    match = (
        "Unsupported operand type(s) for +: 'dict' and 'Transform'\n"
        'The left-side argument must be a length-3 vector.'
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        {} + pv.Transform()


def test_transform_rmul_raises():
    match = (
        "Unsupported operand value(s) for *: 'tuple' and 'Transform'\n"
        'The left-side argument must be a single number or a length-3 vector.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        (1, 2, 3, 4) * pv.Transform()

    match = (
        "Unsupported operand type(s) for *: 'dict' and 'Transform'\n"
        'The left-side argument must be a single number or a length-3 vector.'
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        {} * pv.Transform()


def test_transform_mul_raises():
    match = (
        "Unsupported operand value(s) for *: 'Transform' and 'tuple'\n"
        'The right-side argument must be a single number or a length-3 vector.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.Transform() * (1, 2, 3, 4)

    match = (
        "Unsupported operand type(s) for *: 'Transform' and 'dict'\n"
        'The right-side argument must be a single number or a length-3 vector.'
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        pv.Transform() * {}


def test_transform_matmul_raises():
    match = (
        "Unsupported operand value(s) for @: 'Transform' and 'tuple'\n"
        'The right-side argument must be transform-like.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.Transform() @ (1, 2, 3, 4)

    match = (
        "Unsupported operand type(s) for @: 'Transform' and 'dict'\n"
        'The right-side argument must be transform-like.'
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        pv.Transform() @ {}


@pytest.mark.parametrize('multiply_mode', ['pre', 'post'])
def test_transform_copy(multiply_mode):
    t1 = Transform().scale(SCALE)
    t1.multiply_mode = multiply_mode
    t2 = t1.copy()
    assert np.array_equal(t1.matrix, t2.matrix)
    assert t1 is not t2
    assert t2.multiply_mode == t1.multiply_mode


def test_transform_repr(transform):
    def _repr_no_first_line(trans):
        return '\n'.join(repr(trans).split('\n')[1:])

    # Test compact format with no unnecessary spacing
    repr_ = _repr_no_first_line(transform)
    assert repr_ == (
        '  Num Transformations: 0\n'
        '  Matrix:  [[1., 0., 0., 0.],\n'
        '            [0., 1., 0., 0.],\n'
        '            [0., 0., 1., 0.],\n'
        '            [0., 0., 0., 1.]]'
    )

    # Test with floats which have many decimals
    transform.concatenate(pv.transformations.axis_angle_rotation((0, 0, 1), 45))
    repr_ = _repr_no_first_line(transform)
    assert repr_ == (
        '  Num Transformations: 1\n'
        '  Matrix:  [[ 0.70710678, -0.70710678,  0.        ,  0.        ],\n'
        '            [ 0.70710678,  0.70710678,  0.        ,  0.        ],\n'
        '            [ 0.        ,  0.        ,  1.        ,  0.        ],\n'
        '            [ 0.        ,  0.        ,  0.        ,  1.        ]]'
    )


values = (0.1, 0.2, 0.3)
SHEAR = np.eye(3)
SHEAR[0, 1] = values[0]
SHEAR[1, 0] = values[0]
SHEAR[0, 2] = values[1]
SHEAR[2, 0] = values[1]
SHEAR[1, 2] = values[2]
SHEAR[2, 1] = values[2]


@pytest.mark.parametrize('do_shear', [True, False])
@pytest.mark.parametrize('do_scale', [True, False])
@pytest.mark.parametrize('do_reflection', [True, False])
@pytest.mark.parametrize('do_rotate', [True, False])
@pytest.mark.parametrize('do_translate', [True, False])
def test_transform_decompose(transform, do_shear, do_scale, do_reflection, do_rotate, do_translate):
    if do_shear:
        transform.concatenate(SHEAR)
    if do_scale:
        transform.scale(VECTOR)
    if do_reflection:
        transform.scale(-1)
    if do_rotate:
        transform.rotate(ROTATION)
    if do_translate:
        transform.translate(VECTOR)

    T, R, N, S, K = transform.decompose()

    assert isinstance(T, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(N, np.ndarray)
    assert isinstance(S, np.ndarray)
    assert isinstance(K, np.ndarray)

    expected_translation = VECTOR if do_translate else np.zeros((3,))
    expected_rotation = ROTATION if do_rotate else np.eye(3)
    expected_reflection = -1 if do_reflection else 1
    expected_scale = VECTOR if do_scale else np.ones((3,))
    expected_shear = SHEAR if do_shear else np.eye(3)

    # Test decomposed translation and reflection always matches input exactly
    assert np.allclose(T, expected_translation)
    assert np.allclose(N, expected_reflection)
    # Test rotation, scale, and shear always matches input exactly unless
    # scale and shear and both specified
    is_exact_decomposition = not (do_scale and do_shear)
    assert np.allclose(R, expected_rotation) == is_exact_decomposition
    assert np.allclose(S, expected_scale) == is_exact_decomposition
    assert np.allclose(K, expected_shear) == is_exact_decomposition

    # Test composition from decomposed elements matches input
    T, R, N, S, K = transform.decompose(homogeneous=True)
    recomposed = pv.Transform([T, R, N, S, K], multiply_mode='pre')
    assert np.allclose(recomposed.matrix, transform.matrix)


@pytest.mark.parametrize('homogeneous', [True, False])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_transform_decompose_dtype(dtype, homogeneous):
    matrix = np.eye(4).astype(dtype)
    T, R, N, S, K = transformations.decomposition(matrix, homogeneous=homogeneous)
    assert np.issubdtype(T.dtype, dtype)
    assert np.issubdtype(R.dtype, dtype)
    assert np.issubdtype(N.dtype, dtype)
    assert np.issubdtype(S.dtype, dtype)
    assert np.issubdtype(K.dtype, dtype)
