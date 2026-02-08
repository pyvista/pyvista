"""Test pyvista core utilities."""

from __future__ import annotations

from collections.abc import Iterable
import contextlib
import inspect
import itertools
import json
import operator
import os
from pathlib import Path
import pickle
import platform
import re
import shutil
import sys
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeVar
from typing import get_args
from unittest import mock
import warnings

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import pytest
from pytest_cases import parametrize
from pytest_cases import parametrize_with_cases
from scipy.spatial.transform import Rotation
from scooby.report import get_distribution_dependencies

import pyvista as pv
from pyvista import examples as ex
from pyvista._deprecate_positional_args import _MAX_POSITIONAL_ARGS
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk
from pyvista.core._vtk_utilities import is_vtk_attribute
from pyvista.core.celltype import _CELL_TYPE_INFO
from pyvista.core.utilities import cells
from pyvista.core.utilities import fileio
from pyvista.core.utilities import fit_line_to_points
from pyvista.core.utilities import fit_plane_to_points
from pyvista.core.utilities import line_segments_from_points
from pyvista.core.utilities import principal_axes
from pyvista.core.utilities import transformations
from pyvista.core.utilities import vector_poly_data
from pyvista.core.utilities.arrays import _coerce_pointslike_arg
from pyvista.core.utilities.arrays import _SerializedDictArray
from pyvista.core.utilities.arrays import convert_array
from pyvista.core.utilities.arrays import copy_vtk_array
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.arrays import get_array_association
from pyvista.core.utilities.arrays import has_duplicates
from pyvista.core.utilities.arrays import parse_field_choice
from pyvista.core.utilities.arrays import raise_has_duplicates
from pyvista.core.utilities.arrays import raise_not_matching
from pyvista.core.utilities.arrays import vtk_id_list_to_array
from pyvista.core.utilities.cell_quality import _CELL_QUALITY_INFO
from pyvista.core.utilities.cell_quality import CellQualityInfo
from pyvista.core.utilities.docs import linkcode_resolve
from pyvista.core.utilities.features import create_grid
from pyvista.core.utilities.features import sample_function
from pyvista.core.utilities.fileio import _CompressionOptions
from pyvista.core.utilities.fileio import get_ext
from pyvista.core.utilities.helpers import is_inside_bounds
from pyvista.core.utilities.misc import AnnotatedIntEnum
from pyvista.core.utilities.misc import _classproperty
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import assert_empty_kwargs
from pyvista.core.utilities.misc import check_valid_vector
from pyvista.core.utilities.misc import has_module
from pyvista.core.utilities.observers import Observer
from pyvista.core.utilities.observers import ProgressMonitor
from pyvista.core.utilities.state_manager import _StateManager
from pyvista.core.utilities.state_manager import _update_alg
from pyvista.core.utilities.transform import Transform
from pyvista.core.utilities.writer import _DataFormatMixin
from pyvista.plotting.prop3d import _orientation_as_rotation_matrix
from pyvista.plotting.widgets import _parse_interaction_event
from tests.conftest import NUMPY_VERSION_INFO

with contextlib.suppress(ImportError):
    import tomllib  # Python 3.11+

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

IS_ARM_MAC = platform.system() == 'Darwin' and platform.machine() == 'arm64'


@pytest.fixture
def transform():
    return Transform()


def test_sample_function_raises(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setattr(os, 'name', 'nt')
        with pytest.raises(
            ValueError,
            match='This function on Windows only supports int32 or smaller',
        ):
            sample_function(_vtk.vtkPlane(), output_type=np.int64)

        with pytest.raises(
            ValueError,
            match='This function on Windows only supports int32 or smaller',
        ):
            sample_function(_vtk.vtkPlane(), output_type=np.uint64)

        with pytest.raises(
            ValueError,
            match='Invalid output_type 1',
        ):
            sample_function(_vtk.vtkPlane(), output_type=1)


def test_progress_monitor_raises(mocker: MockerFixture):
    from pyvista.core.utilities import observers

    m = mocker.patch.object(observers, 'importlib')
    m.util.find_spec.return_value = False

    with pytest.raises(
        ImportError,
        match=r'Please install `tqdm` to monitor algorithms.',
    ):
        ProgressMonitor('algo')


def test_create_grid_raises():
    with pytest.raises(NotImplementedError, match=r'Please specify dimensions.'):
        create_grid(pv.Sphere(), dimensions=None)


def test_parse_field_choice_raises():
    with pytest.raises(ValueError, match=re.escape('Data field (foo) not supported.')):
        parse_field_choice('foo')

    with pytest.raises(TypeError, match=re.escape('Data field (1) not supported.')):
        parse_field_choice(1)


def test_convert_array_raises():
    with pytest.raises(TypeError, match=re.escape("Invalid input array type (<class 'int'>).")):
        convert_array(1)


def test_get_array_raises():
    with pytest.raises(
        KeyError, match=re.escape("'Data array (foo) not present in this dataset.'")
    ):
        get_array(_vtk.vtkTable(), 'foo', err=True)

    with pytest.raises(
        KeyError, match=re.escape("'Data array (foo) not present in this dataset.'")
    ):
        get_array_association(_vtk.vtkTable(), 'foo', err=True)


def test_raise_not_matching_raises():
    with pytest.raises(
        ValueError,
        match=re.escape('Number of scalars (1) must match number of rows (0).'),
    ):
        raise_not_matching(scalars=np.array([0.0]), dataset=pv.Table())


def test_vtk_version_info():
    ver = _vtk.vtkVersion()
    assert ver.GetVTKMajorVersion() == pv.vtk_version_info.major
    assert ver.GetVTKMinorVersion() == pv.vtk_version_info.minor
    assert ver.GetVTKBuildVersion() == pv.vtk_version_info.micro
    ver_tup = (
        ver.GetVTKMajorVersion(),
        ver.GetVTKMinorVersion(),
        ver.GetVTKBuildVersion(),
    )
    assert str(ver_tup) == str(pv.vtk_version_info)
    assert ver_tup == pv.vtk_version_info
    assert pv.vtk_version_info >= pv._MIN_SUPPORTED_VTK_VERSION


@pytest.mark.parametrize('operation', [operator.le, operator.lt, operator.gt, operator.ge])
def test_vtk_version_info_raises(operation):
    version_str = '.'.join(map(str, pv._MIN_SUPPORTED_VTK_VERSION))
    match = f'Comparing against unsupported VTK version 1.2.3. Minimum supported is {version_str}'
    with pytest.raises(pv.VTKVersionError, match=match):
        operation(pv.vtk_version_info, (1, 2, 3))


@pytest.mark.skipif(
    sys.version_info < (3, 11) or sys.platform == 'darwin',
    reason='Requires Python 3.11+, path issues on macOS',
)
def test_min_supported_vtk_version_matches_pyproject():
    def get_min_vtk_version_from_pyproject():
        # locate pyproject.toml relative to package
        root = Path(
            os.environ.get('TOX_ROOT', Path(pv.__file__).parents[1])
        )  # to make the test work when pyvista is installed via tox
        pyproject_path = root / 'pyproject.toml'

        with pyproject_path.open('rb') as f:
            pyproject_data = tomllib.load(f)

        # dependencies live under [project]
        dependencies = pyproject_data.get('project', {}).get('dependencies', [])

        # find the first vtk>= spec and strip env markers after `;`
        min_vtk = next(
            dep.split(';', 1)[0].strip().split('>=', 1)[1]
            for dep in dependencies
            if 'vtk>=' in dep.split(';', 1)[0]
        )
        assert isinstance(min_vtk, str)
        assert len(min_vtk) > 0
        return tuple(map(int, min_vtk.split('.')))

    from_pyproject = get_min_vtk_version_from_pyproject()
    from_code = pv._MIN_SUPPORTED_VTK_VERSION
    msg = (
        f"Min VTK version specified in 'pyproject.toml' should match the "
        f'min version specified in {_vtk.__name__!r}'
    )
    assert from_pyproject == from_code, msg


def test_createvectorpolydata_error():
    orig = np.random.default_rng().random((3, 1))
    with pytest.raises(ValueError, match='orig array must be 3D'):
        vector_poly_data(orig, [0, 1, 2])

    vec = np.random.default_rng().random((3, 1))
    with pytest.raises(ValueError, match='vec array must be 3D'):
        vector_poly_data([0, 1, 2], vec)


def test_createvectorpolydata_1d():
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
    fnames = (
        ex.antfile,
        ex.planefile,
        ex.hexbeamfile,
        ex.spherefile,
        ex.uniformfile,
        ex.rectfile,
    )
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
    fnames = (
        ex.antfile,
        ex.planefile,
        ex.hexbeamfile,
        ex.spherefile,
        ex.uniformfile,
        ex.rectfile,
    )
    types = (
        pv.PolyData,
        pv.PolyData,
        pv.UnstructuredGrid,
        pv.PolyData,
        pv.ImageData,
        pv.RectilinearGrid,
    )

    dummy_extension = '.dummy'
    for fname, type_ in zip(fnames, types, strict=True):
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
def test_read_progress_bar(mock_show_progress, mock_reader, mock_read):  # noqa: ARG001
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
    args, _kwargs = read_exodus_mock.call_args
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
    grid_vtk = _vtk.vtkUnstructuredGrid()
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


def test_is_inside_bounds_raises():
    with pytest.raises(ValueError, match='Bounds mismatch point dimensionality'):
        is_inside_bounds(point=np.array([0]), bounds=(0,))

    with pytest.raises(ValueError, match='Bounds mismatch point dimensionality'):
        is_inside_bounds(point=np.array([0]), bounds=(0, 1, 3, 4, 5))

    with pytest.raises(
        TypeError, match=re.escape("Unknown input data type (<class 'NoneType'>).")
    ):
        is_inside_bounds(point=None, bounds=(0,))


def test_voxelize(uniform):
    with pytest.warns(pv.PyVistaDeprecationWarning):
        vox = pv.voxelize(uniform, density=0.5)
    assert vox.n_cells

    if pv._version.version_info[:2] > (0, 49):
        msg = 'Remove this deprecated function.'
        raise RuntimeError(msg)


def test_voxelize_non_uniform_density(uniform):
    with pytest.warns(pv.PyVistaDeprecationWarning):
        vox = pv.voxelize(uniform, density=[0.5, 0.3, 0.2])
    assert vox.n_cells
    with pytest.warns(pv.PyVistaDeprecationWarning):
        vox = pv.voxelize(uniform, density=np.array([0.5, 0.3, 0.2]))
    assert vox.n_cells


def test_voxelize_invalid_density(rectilinear):
    # test error when density is not length-3
    with pytest.warns(pv.PyVistaDeprecationWarning):
        with pytest.raises(ValueError, match='not enough values to unpack'):
            pv.voxelize(rectilinear, density=[0.5, 0.3])
    # test error when density is not an array-like
    with pytest.warns(pv.PyVistaDeprecationWarning):
        with pytest.raises(TypeError, match='expected number or array-like'):
            pv.voxelize(rectilinear, density={0.5, 0.3})


def test_voxelize_throws_point_cloud(hexbeam):
    mesh = pv.PolyData(hexbeam.points)
    with pytest.warns(pv.PyVistaDeprecationWarning):
        with pytest.raises(ValueError, match='must have faces'):
            pv.voxelize(mesh)


def test_voxelize_volume_default_density(uniform):
    with pytest.warns(pv.PyVistaDeprecationWarning):
        expected = pv.voxelize_volume(uniform, density=uniform.length / 100).n_cells
    with pytest.warns(pv.PyVistaDeprecationWarning):
        actual = pv.voxelize_volume(uniform).n_cells
    assert actual == expected

    if pv._version.version_info[:2] > (0, 49):
        msg = 'Remove this deprecated function.'
        raise RuntimeError(msg)


def test_voxelize_volume_invalid_density(rectilinear):
    with pytest.warns(pv.PyVistaDeprecationWarning):
        with pytest.raises(TypeError, match='expected number or array-like'):
            pv.voxelize_volume(rectilinear, density={0.5, 0.3})


def test_voxelize_volume_no_face_mesh(rectilinear):
    with pytest.warns(pv.PyVistaDeprecationWarning):
        with pytest.raises(ValueError, match='must have faces'):
            pv.voxelize_volume(pv.PolyData())
    with pytest.warns(pv.PyVistaDeprecationWarning):
        with pytest.raises(TypeError, match='expected number or array-like'):
            pv.voxelize_volume(rectilinear, density={0.5, 0.3})


@pytest.mark.parametrize('function', [pv.voxelize_volume, pv.voxelize])
def test_voxelize_enclosed_bounds(function, ant):
    with pytest.warns(pv.PyVistaDeprecationWarning):
        vox = function(ant, density=0.9, enclosed=True)

    assert vox.bounds.x_min <= ant.bounds.x_min
    assert vox.bounds.y_min <= ant.bounds.y_min
    assert vox.bounds.z_min <= ant.bounds.z_min

    assert vox.bounds.x_max >= ant.bounds.x_max
    assert vox.bounds.y_max >= ant.bounds.y_max
    assert vox.bounds.z_max >= ant.bounds.z_max


@pytest.mark.parametrize('function', [pv.voxelize_volume, pv.voxelize])
def test_voxelize_fit_bounds(function, uniform):
    with pytest.warns(pv.PyVistaDeprecationWarning):
        vox = function(uniform, density=0.9, fit_bounds=True)

    assert np.isclose(vox.bounds.x_min, uniform.bounds.x_min)
    assert np.isclose(vox.bounds.y_min, uniform.bounds.y_min)
    assert np.isclose(vox.bounds.z_min, uniform.bounds.z_min)

    assert np.isclose(vox.bounds.x_max, uniform.bounds.x_max)
    assert np.isclose(vox.bounds.y_max, uniform.bounds.y_max)
    assert np.isclose(vox.bounds.z_max, uniform.bounds.z_max)


@pytest.mark.skip_catch_vtk_errors
def test_report():
    report = pv.Report(gpu=True)
    assert report is not None
    assert 'GPU Details : None' not in report.__repr__()
    assert re.search(r'Render Window : (vtk\w+RenderWindow|error)', report.__repr__())
    assert 'vtkRenderWindow' not in report.__repr__()  # must not be abstract
    report = pv.Report(gpu=False)
    assert report is not None
    assert 'GPU Details : None' in report.__repr__()
    assert 'Render Window : None' in report.__repr__()
    assert 'User Data Path' not in report.__repr__()


def test_report_warnings():
    with pytest.warns(pv.PyVistaDeprecationWarning):
        pv.Report('vtk', 4, 90, True)


REPORT = str(pv.Report(gpu=False))


@pytest.mark.parametrize('package', get_distribution_dependencies('pyvista'))
def test_report_dependencies(package):
    if package == 'pyvista[colormaps,io,jupyter]':
        pytest.xfail('scooby bug: https://github.com/banesullivan/scooby/issues/129')
    elif package == 'vtk!':
        pytest.xfail('scooby bug: https://github.com/banesullivan/scooby/issues/133')
    elif package == 'jupyter-server-proxy':
        pytest.xfail('not installed with --test group')
    assert package in REPORT


def test_report_downloads():
    report = pv.Report(downloads=True)
    repr_ = repr(report)
    assert f'User Data Path : {pv.examples.downloads.USER_DATA_PATH}' in repr_
    assert f'VTK Data Source : {pv.examples.downloads.SOURCE}' in repr_
    assert f'File Cache : {pv.examples.downloads._FILE_CACHE}' in repr_


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
    uu, vv, ww = pv.transform_vectors_sph_to_cart(theta=lon, phi=lat, r=lev, u=u, v=v, w=w)
    assert np.allclose(
        [uu[-1, -1], vv[-1, -1], ww[-1, -1]],
        [67.80403533828323, 360.8359915416445, -70000.0],
    )


def test_vtkmatrix_to_from_array():
    rng = np.random.default_rng()
    array3x3 = rng.integers(0, 10, size=(3, 3))
    matrix = pv.vtkmatrix_from_array(array3x3)
    assert isinstance(matrix, _vtk.vtkMatrix3x3)
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
    assert isinstance(matrix, _vtk.vtkMatrix4x4)
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
    invalid = _vtk.vtkTransform()
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
    id_list = _vtk.vtkIdList()
    id_list.SetNumberOfIds(len(ids))
    for i, v in enumerate(ids):
        id_list.SetId(i, v)
    converted = vtk_id_list_to_array(id_list)
    assert np.allclose(converted, ids)


def test_progress_monitor():
    mesh = pv.Sphere()
    ugrid = mesh.warp_by_vector(progress_bar=True)
    assert isinstance(ugrid, pv.PolyData)


def test_observer():
    msg = 'KIND: In PATH, line 0\nfoo (0x000000): ALERT'
    obs = Observer()
    ret = obs.parse_message('foo')
    assert ret[3] == 'foo'
    ret = obs.parse_message(msg)
    assert ret[0] == 'KIND' == ret.kind
    assert ret[1] == 'PATH' == ret.path
    assert ret[2] == '0x000000' == ret.address
    assert ret[4] == '0' == ret.line
    assert ret[5] == 'foo' == ret.name
    assert ret[3] == 'ALERT' == ret.alert
    assert str(ret) == msg

    for kind in ['WARNING', 'ERROR']:
        obs.log_message(kind, 'foo')

    # Pass positionally as that's what VTK will do
    obs(None, None, msg)
    assert obs.has_event_occurred()
    assert obs.get_message() == 'ALERT'
    assert obs.get_message(etc=True) == msg

    alg = _vtk.vtkSphereSource()
    alg.GetExecutive()
    obs.observe(alg)
    with pytest.raises(RuntimeError, match='algorithm'):
        obs.observe(alg)


def test_observer_default_event():
    msg = 'This message does not match parsing regex!'
    obs = Observer()
    ret = obs.parse_message(msg)
    assert ret.kind == ''
    assert ret.path == ''
    assert ret.address == ''
    assert ret.line == ''
    assert ret.name == ''
    assert ret.alert == msg

    assert str(ret) == msg


@pytest.mark.parametrize('point', [1, object(), None])
def test_valid_vector_raises(point):
    with pytest.raises(TypeError, match=r'foo must be a length three iterable of floats.'):
        check_valid_vector(point=point, name='foo')


def test_check_valid_vector():
    with pytest.raises(ValueError, match='length three'):
        check_valid_vector([0, 1])

    check_valid_vector([0, 1, 2])


@pytest.mark.parametrize('value', [object(), None, [], ()])
def test_annotated_int_enum_from_any_raises(value):
    class Foo(AnnotatedIntEnum):
        BAR = (0, 'foo')

    with pytest.raises(
        TypeError,
        match=re.escape(f'Invalid type {type(value)} for class {Foo.__name__}'),
    ):
        Foo.from_any(value)


@given(points=st.lists(st.integers()).filter(lambda x: bool(len(x) % 2)))
def test_lines_segments_from_points(points):
    with pytest.raises(
        ValueError,
        match=r'An even number of points must be given to define each segment.',
    ):
        line_segments_from_points(points=points)


def test_cells_dict_utils():
    # No pyvista object
    with pytest.raises(TypeError):
        cells.get_mixed_cells(None)

    with pytest.raises(TypeError):
        cells.get_mixed_cells(np.zeros(shape=[3, 3]))


@pytest.mark.skipif(
    NUMPY_VERSION_INFO < (2, 3) and IS_ARM_MAC,
    reason='Specific to Mac M4. See https://github.com/numpy/numpy/issues/28687',
)
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
    from vtkmodules.vtkIOLegacy import vtkDataWriter

    # vtkWriter.cxx:55     ERR| vtkDataWriter (0x141efbd10): No input provided!
    writer = vtkDataWriter()
    writer.Write()


def _generate_vtk_warn():
    """Simple operation which generates a VTK warning."""
    from vtkmodules.vtkFiltersCore import vtkMergeFilter

    # vtkMergeFilter.cxx:277   WARN| vtkMergeFilter (0x600003c18000): Nothing to merge!
    merge = vtkMergeFilter()
    merge.AddInputData(_vtk.vtkPolyData())
    merge.Update()


def test_vtk_error_catcher():
    # raise_errors: False
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher()
    with error_catcher:
        _generate_vtk_err()
        _generate_vtk_err()
        _generate_vtk_warn()
        _generate_vtk_warn()
    assert len(error_catcher.events) == 4
    assert len(error_catcher.error_events) == 2
    assert len(error_catcher.warning_events) == 2

    # raise_errors: False, no error
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher()
    with error_catcher:
        pass

    # raise_errors: True
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher(raise_errors=True)
    error_match = re.compile(
        r'ERROR: In vtkWriter\.cxx, line \d+\n'
        r'vtkDataWriter \(0x?[0-9a-fA-F]+\): No input provided!\n*'
    )
    with pytest.raises(RuntimeError, match=re.compile(error_match)):  # noqa: PT012
        with error_catcher:
            _generate_vtk_err()
            _generate_vtk_warn()
    assert len(error_catcher.events) == 2
    assert len(error_catcher.error_events) == 1
    assert len(error_catcher.warning_events) == 1

    # Raise two VTK errors as a single RuntimeError
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher(
        raise_errors=True, emit_warnings=True
    )
    error_match2 = re.compile(f'{error_match.pattern}\n{error_match.pattern}')
    with pytest.raises(pv.VTKExecutionError, match=error_match2):  # noqa: PT012
        with error_catcher:
            _generate_vtk_err()
            _generate_vtk_err()

    # Warn and raise error. The order emitted by VTK is not guaranteed to be the same
    # since warn and err events are logged independently.
    # Here we generate VTK err then warn, but expect warning then error
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher(
        raise_errors=True, emit_warnings=True
    )
    warning_match = re.compile(
        r'Warning: In vtkMergeFilter\.cxx, line \d+\n'
        r'vtkMergeFilter \(0x?[0-9a-fA-F]+\): Nothing to merge!'
    )
    with pytest.warns(pv.VTKExecutionWarning, match=warning_match):  # noqa: PT031
        with pytest.raises(pv.VTKExecutionError, match=error_match):  # noqa: PT012
            with error_catcher:
                _generate_vtk_err()
                _generate_vtk_warn()

    # Test raise/emit with no errors/warnings generated
    error_catcher = pv.core.utilities.observers.VtkErrorCatcher(
        raise_errors=True, emit_warnings=True
    )
    with error_catcher:
        pass


@pytest.mark.skip_catch_vtk_errors
def test_update_alg_raises():
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    reader = vtkXMLPolyDataReader()

    reader.SetFileName('this_file_does_not_exist.vtp')
    with pv.vtk_message_policy('off'):
        with pytest.raises(pv.VTKExecutionError):
            _update_alg(reader)


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

    with pytest.raises(TypeError, match=r'Expected pyvista.DataSet'):
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
    assert merged_ugrid.n_points == sum(ds.n_points for ds in datasets)
    # check main has priority
    sphere_main = sphere.copy()
    sphere_other = sphere.copy()
    main_data = np.zeros(sphere_main.n_points)
    other_data = np.ones(sphere_main.n_points)
    sphere_main['data'] = main_data
    sphere_other['data'] = other_data

    merged = pv.merge(
        [sphere_main, sphere_other],
        merge_points=True,
    )
    assert np.allclose(merged['data'], main_data)


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
    arr = _vtk.vtkFloatArray()
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


def test_copy_implicit_vtk_array(plane):
    # Use the connectivity filter to generate an implicit vtkDataArray
    conn = plane.connectivity()
    vtk_object = conn['RegionId'].VTKObject
    if pv.vtk_version_info >= (9, 4):
        # The VTK array appears to be abstract but is not
        assert type(vtk_object) is _vtk.vtkDataArray
    else:
        assert type(vtk_object) is _vtk.vtkIdTypeArray

    # `copy_vtk_array` is called with this assignment
    plane['test'] = conn['RegionId']

    new_vtk_object = plane['test'].VTKObject
    if pv.vtk_version_info >= (9, 4):
        # The VTK array type has changed and is now a concrete subclass
        assert type(new_vtk_object) is _vtk.vtkTypeInt64Array
    else:
        assert type(new_vtk_object) is _vtk.vtkIdTypeArray


def test_cartesian_to_spherical():
    def polar2cart(r, phi, theta):
        return np.vstack(
            (
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ),
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

    # test wrapped function
    link = linkcode_resolve(
        'py',
        {'module': 'pyvista', 'fullname': 'pyvista.plotting.plotter.Plotter.add_ruler'},
    )
    assert 'renderer.py' in link

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

    fitted_line = fit_line_to_points(
        expected_line.points, resolution=resolution, return_meta=False
    )
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


@pytest.mark.skipif(
    NUMPY_VERSION_INFO < (1, 26) or IS_ARM_MAC,
    reason='Different results for some tests.',
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


@pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:invalid value encountered in divide:RuntimeWarning')
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
def no_new_attributes_mixin_subclass():
    class A(_NoNewAttrMixin):
        def __init__(self):
            super().__init__()
            self.bar = 42

    class B(A):
        def __init__(self):
            super().__init__()
            self.baz = 42

    return A(), B()


def test_no_new_attr_mixin(no_new_attributes_mixin_subclass):
    a, b = no_new_attributes_mixin_subclass
    ham = 'ham'
    eggs = 'eggs'

    match = (
        "Attribute 'ham' does not exist and cannot be added to class 'A'\nUse "
        '`pyvista.set_new_attribute` or `pyvista.allow_new_attributes` to set new attributes.\n'
        'Setting new private variables (with `_` prefix) is allowed by default.'
    )
    with pytest.raises(pv.PyVistaAttributeError, match=re.escape(match)):
        setattr(a, ham, eggs)

    match = "Attribute 'ham' does not exist and cannot be added to class 'B'"
    with pytest.raises(pv.PyVistaAttributeError, match=match):
        setattr(b, ham, eggs)


def test_set_new_attribute(no_new_attributes_mixin_subclass):
    a, _ = no_new_attributes_mixin_subclass
    ham = 'ham'
    eggs = 'eggs'

    assert not hasattr(a, ham)
    pv.set_new_attribute(a, ham, eggs)
    assert hasattr(a, ham)
    assert getattr(a, ham) == eggs

    match = (
        "Attribute 'ham' already exists. "
        '`set_new_attribute` can only be used for setting NEW attributes.'
    )
    with pytest.raises(pv.PyVistaAttributeError, match=re.escape(match)):
        pv.set_new_attribute(a, ham, eggs)


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
    assert str(serial_dict) == '{}'

    # init from dict
    new_dict = dict(ham='eggs')
    serial_dict = _SerializedDictArray(new_dict)
    assert serial_dict['ham'] == 'eggs'
    assert str(serial_dict) == '{"ham": "eggs"}'

    # init from UserDict
    serial_dict = _SerializedDictArray(serial_dict)
    assert serial_dict['ham'] == 'eggs'
    assert str(serial_dict) == '{"ham": "eggs"}'

    # init from JSON string
    json_dict = json.dumps(new_dict)
    serial_dict = _SerializedDictArray(json_dict)
    assert serial_dict['ham'] == 'eggs'
    assert str(serial_dict) == '{"ham": "eggs"}'


def test_serial_dict_as_dict(serial_dict_with_foobar):
    assert not isinstance(serial_dict_with_foobar, dict)
    actual_dict = dict(serial_dict_with_foobar)
    assert isinstance(actual_dict, dict)
    assert actual_dict == serial_dict_with_foobar.data


def test_serial_dict_overrides__setitem__(serial_dict_empty):
    serial_dict_empty['foo'] = 'bar'
    assert str(serial_dict_empty) == '{"foo": "bar"}'


def test_serial_dict_overrides__delitem__(serial_dict_with_foobar):
    del serial_dict_with_foobar['foo']
    assert str(serial_dict_with_foobar) == '{}'


def test_serial_dict_overrides__setattr__(serial_dict_empty):
    serial_dict_empty.data = dict(foo='bar')
    assert str(serial_dict_empty) == '{"foo": "bar"}'


def test_serial_dict_overrides_popitem(serial_dict_with_foobar):
    serial_dict_with_foobar['ham'] = 'eggs'
    item = serial_dict_with_foobar.popitem()
    assert item == ('foo', 'bar')
    assert str(serial_dict_with_foobar) == '{"ham": "eggs"}'


def test_serial_dict_overrides_pop(serial_dict_with_foobar):
    item = serial_dict_with_foobar.pop('foo')
    assert item == 'bar'
    assert str(serial_dict_with_foobar) == '{}'


def test_serial_dict_overrides_update(serial_dict_empty):
    serial_dict_empty.update(dict(foo='bar'))
    assert str(serial_dict_empty) == '{"foo": "bar"}'


def test_serial_dict_overrides_clear(serial_dict_with_foobar):
    serial_dict_with_foobar.clear()
    assert str(serial_dict_with_foobar) == '{}'


def test_serial_dict_overrides_setdefault(serial_dict_empty, serial_dict_with_foobar):
    serial_dict_empty.setdefault('foo', 42)
    assert str(serial_dict_empty) == '{"foo": 42}'
    serial_dict_with_foobar.setdefault('foo', 42)
    assert str(serial_dict_with_foobar) == '{"foo": "bar"}'


SCALE = 2
ROTATION = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # rotate 90 deg about z axis
VECTOR = (1, 2, 3)
ANGLE = 30


@pytest.mark.parametrize('scale_args', [(SCALE,), (SCALE, SCALE, SCALE), [(SCALE, SCALE, SCALE)]])
def test_transform_scale(transform, scale_args):
    assert not transform.has_scale
    transform.scale(*scale_args)
    actual = transform.matrix
    expected = np.diag((SCALE, SCALE, SCALE, 1))
    assert np.array_equal(actual, expected)
    assert transform.n_transformations == 1
    assert transform.has_scale
    assert np.array_equal(transform.scale_factors, (SCALE, SCALE, SCALE))

    identity = transform.matrix @ transform.inverse_matrix
    assert np.array_equal(identity, np.eye(4))


@pytest.mark.parametrize('translate_args', [np.array(VECTOR), np.array([VECTOR])])
def test_transform_translate(transform, translate_args):
    assert not transform.has_translation
    transform.translate(*translate_args)
    actual = transform.matrix
    expected = np.eye(4)
    expected[:3, 3] = VECTOR
    assert np.array_equal(actual, expected)
    assert transform.has_translation
    assert np.array_equal(transform.translation, VECTOR)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.array_equal(identity, np.eye(4))


@pytest.mark.parametrize('reflect_args', [VECTOR, [VECTOR]])
def test_transform_reflect(transform, reflect_args):
    assert transform.reflection == 1
    assert not transform.has_reflection
    transform.reflect(*reflect_args)
    actual = transform.matrix
    expected = transformations.reflection(VECTOR)
    assert np.array_equal(actual, expected)
    assert transform.has_reflection
    assert transform.reflection == -1

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


@pytest.mark.parametrize(
    ('method', 'vector'),
    [('flip_x', (1, 0, 0)), ('flip_y', (0, 1, 0)), ('flip_z', (0, 0, 1))],
)
def test_transform_flip_xyz(transform, method, vector):
    getattr(transform, method)()
    actual = transform.matrix
    expected = transformations.reflection(vector)
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_rotate(transform):
    assert not transform.has_rotation
    transform.rotate(ROTATION)
    actual = transform.matrix
    expected = np.eye(4)
    expected[:3, :3] = ROTATION
    assert np.array_equal(actual, expected)
    assert transform.has_rotation
    assert np.array_equal(transform.rotation_matrix, ROTATION)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.array_equal(identity, np.eye(4))


@pytest.mark.parametrize('multiply_mode', ['post', 'pre'])
@pytest.mark.parametrize(
    ('method', 'args'),
    [
        ('compose', (pv.Transform(ROTATION).translate(VECTOR),)),
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
    axis, angle = transform.rotation_axis_angle
    assert np.allclose(axis, (1, 0, 0))
    assert np.allclose(angle, ANGLE)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_rotate_y(transform):
    transform.rotate_y(ANGLE)
    actual = transform.matrix
    expected = transformations.axis_angle_rotation((0, 1, 0), ANGLE)
    assert np.array_equal(actual, expected)
    axis, angle = transform.rotation_axis_angle
    assert np.allclose(axis, (0, 1, 0))
    assert np.allclose(angle, ANGLE)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_rotate_z(transform):
    transform.rotate_z(ANGLE)
    actual = transform.matrix
    expected = transformations.axis_angle_rotation((0, 0, 1), ANGLE)
    assert np.array_equal(actual, expected)
    axis, angle = transform.rotation_axis_angle
    assert np.allclose(axis, (0, 0, 1))
    assert np.allclose(angle, ANGLE)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_rotate_vector(transform):
    transform.rotate_vector(VECTOR, ANGLE)
    actual = transform.matrix
    expected = transformations.axis_angle_rotation(VECTOR, ANGLE)
    assert np.array_equal(actual, expected)

    identity = transform.matrix @ transform.inverse_matrix
    assert np.allclose(identity, np.eye(4))


def test_transform_compose_vtkmatrix(transform):
    scale_array = np.diag((1, 2, 3, 1))
    vtkmatrix = pv.vtkmatrix_from_array(scale_array)
    transform.compose(vtkmatrix)
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


class CasesTransformApply:
    def case_list_int(self):
        return list(VECTOR), False, np.ndarray, float

    def case_tuple_int(self):
        return VECTOR, False, np.ndarray, float

    def case_array1d_int(self):
        return np.array(VECTOR), False, np.ndarray, float

    def case_array2d_int(self):
        return np.array([VECTOR]), False, np.ndarray, float

    def case_array1d_float(self):
        return np.array(VECTOR, dtype=float), True, np.ndarray, float

    def case_array2d_float(self):
        return np.array([VECTOR], dtype=float), True, np.ndarray, float

    @pytest.mark.filterwarnings('ignore:Points is not a float type.*:UserWarning')
    def case_polydata_float32(self):
        return pv.PolyData(np.atleast_2d(VECTOR)), True, pv.PolyData, np.float32

    @pytest.mark.filterwarnings('ignore:Points is not a float type.*:UserWarning')
    def case_polydata_int(self):
        return (
            pv.PolyData(np.atleast_2d(VECTOR).astype(int)),
            True,
            pv.PolyData,
            np.float32,
        )

    def case_polydata_float(self):
        return (
            pv.PolyData(np.atleast_2d(VECTOR).astype(float)),
            True,
            pv.PolyData,
            float,
        )

    def case_multiblock_float(self):
        return (
            pv.MultiBlock([pv.PolyData(np.atleast_2d(VECTOR).astype(float))]),
            True,
            pv.MultiBlock,
            float,
        )


@parametrize(copy=[True, False])
@parametrize_with_cases(
    ('obj', 'return_self', 'return_type', 'return_dtype'), cases=CasesTransformApply
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
    out = transform.scale(SCALE).apply(obj, copy=copy)

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


@pytest.fixture
def scale_transform():
    return Transform() * SCALE


@pytest.fixture
def translate_transform():
    return Transform() + VECTOR


@pytest.mark.parametrize('method', [pv.Transform.apply, pv.Transform.apply_to_points])
@pytest.mark.parametrize('transformation', ['scale', 'translate'])
def test_transform_apply_to_points(scale_transform, translate_transform, method, transformation):
    array = np.array((0.0, 0.0, 1.0))

    if transformation == 'scale':
        trans = scale_transform
        expected = array * SCALE
    else:
        trans = translate_transform
        expected = array + VECTOR

    if method == pv.Transform.apply:
        transformed = method(trans, array, 'points')
    else:
        transformed = method(trans, array)
    assert np.allclose(transformed, expected)


@pytest.mark.parametrize('method', [pv.Transform.apply, pv.Transform.apply_to_vectors])
@pytest.mark.parametrize('transformation', ['scale', 'translate'])
def test_transform_apply_to_vectors(scale_transform, translate_transform, method, transformation):
    array = np.array((0.0, 0.0, 1.0))

    if transformation == 'scale':
        trans = scale_transform
        expected = array * SCALE
    else:
        trans = translate_transform
        expected = array

    if method == pv.Transform.apply:
        transformed = method(trans, array, 'vectors')
    else:
        transformed = method(trans, array)
    assert np.allclose(transformed, expected)


@pytest.mark.parametrize('mode', ['active_vectors', 'all_vectors'])
@pytest.mark.parametrize('method', [pv.Transform.apply, pv.Transform.apply_to_dataset])
def test_transform_apply_to_dataset(scale_transform, mode, method):
    vector = np.array(VECTOR, dtype=float)
    mesh = pv.PolyData(vector)
    mesh['vector'] = [vector]

    expected = mesh['vector']
    if mode == 'all_vectors':
        expected = expected * SCALE

    transformed = method(scale_transform, mesh, mode)
    assert np.allclose(transformed['vector'], expected)


@pytest.mark.parametrize('mode', ['replace', 'pre-multiply', 'post-multiply'])
@pytest.mark.parametrize('method', [pv.Transform.apply, pv.Transform.apply_to_actor])
def test_transform_apply_to_actor(scale_transform, translate_transform, mode, method):
    expected_matrix = scale_transform.matrix
    actor = pv.Actor()

    transformed = method(scale_transform, actor, mode)
    assert np.allclose(transformed.user_matrix, expected_matrix)

    # Transform again
    transformed = method(translate_transform, transformed, mode)
    if mode == 'replace':
        expected_matrix = translate_transform.matrix
    else:
        expected_matrix = scale_transform.compose(
            translate_transform, multiply_mode=mode.split('-')[0]
        ).matrix
    assert np.allclose(transformed.user_matrix, expected_matrix)


def test_transform_apply_invalid_mode():
    mesh = pv.PolyData()
    array = np.ndarray(())
    actor = pv.Actor()
    trans = pv.Transform()

    match = (
        "Transformation mode 'points' is not supported for datasets. Mode must be one of\n"
        "['active_vectors', 'all_vectors', None]"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        trans.apply(mesh, 'points')

    match = (
        "Transformation mode 'all_vectors' is not supported for arrays. Mode must be one of\n"
        "['points', 'vectors', None]"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        trans.apply(array, 'all_vectors')

    match = (
        "Transformation mode 'vectors' is not supported for actors. Mode must be one of\n"
        "['replace', 'pre-multiply', 'post-multiply', None]"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        trans.apply(actor, 'vectors')


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


@pytest.mark.parametrize('point', [None, VECTOR])
def test_transform_set_matrix(point):
    # Create transform using point
    trans = pv.Transform(point=point).scale(SCALE)
    new_matrix = pv.Transform(ROTATION).matrix
    assert not np.allclose(trans.matrix, new_matrix)
    trans.matrix = new_matrix
    assert np.allclose(trans.matrix, new_matrix)


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
def test_transform_multiply_mode_override(
    transform, transformed_actor, object_mode, override_mode
):
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
    transform.compose(transformed_actor.user_matrix, multiply_mode=override_mode)

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
    assert np.allclose(transform.matrix, matrix)

    transform = Transform(matrix.tolist())
    assert np.allclose(transform.matrix, matrix)

    transform = Transform(matrix.tolist())
    assert np.array_equal(transform.matrix, matrix)


def test_transform_equivalent_methods():
    def assert_transform_equivalence(tr_a: pv.Transform, tr_b: pv.Transform):
        A = (tr_a * tr_b.inverse_matrix).matrix
        B = np.eye(4)
        assert np.allclose(A, B)
        assert tr_a.n_transformations == tr_b.n_transformations

    # All these transformations should be the same
    tr1 = pv.Transform(ROTATION, point=VECTOR)
    tr2 = pv.Transform(point=VECTOR).rotate(ROTATION)
    tr3 = pv.Transform().rotate(ROTATION, point=VECTOR)
    tr4 = pv.Transform().translate(-np.array(VECTOR)).rotate(ROTATION).translate(VECTOR)

    trans = [tr1, tr2, tr3, tr4]

    for _tr1, _tr2 in itertools.combinations(trans, 2):
        assert_transform_equivalence(_tr1, _tr2)


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
        .compose(eye4)
        .invert()
        .post_multiply()
        .pre_multiply()
        .matrix
    )
    assert np.array_equal(matrix, eye4)


def test_transform_mul():
    scale = Transform().scale(SCALE)
    translate = Transform().translate(VECTOR)

    transform = pv.Transform().post_multiply().translate(VECTOR).scale(SCALE)
    transform_add = translate * scale
    assert np.array_equal(transform_add.matrix, transform.matrix)

    # Validate with numpy matmul
    matrix_numpy = scale.matrix @ translate.matrix
    assert np.array_equal(transform_add.matrix, matrix_numpy)


def test_transform_add():
    transform_base = pv.Transform().post_multiply().scale(SCALE)
    # Translate with `translate` and `+`
    transform_translate = transform_base.copy().translate(VECTOR)
    transform_add = transform_base + VECTOR
    assert np.array_equal(transform_add.matrix, transform_translate.matrix)

    # Test multiply mode override to ensure post-multiply is always used
    transform_add = transform_base.pre_multiply() + VECTOR
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


@pytest.mark.parametrize(
    'other',
    [
        SCALE,
        (SCALE, SCALE, SCALE),
        Transform().scale(SCALE),
        Transform().scale(SCALE).matrix,
    ],
)
def test_transform_mul_other(other):
    transform_base = pv.Transform().post_multiply().translate(VECTOR)
    # Scale with `scale` and `*`
    transform_scale = transform_base.copy().scale(SCALE)
    transform_mul = transform_base * other
    assert np.array_equal(transform_mul.matrix, transform_scale.matrix)

    # Test multiply mode override to ensure post-multiply is always used
    transform_mul = transform_base.pre_multiply() * other
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


def test_transform_add_raises():
    match = (
        "Unsupported operand value(s) for +: 'Transform' and 'int'\n"
        'The right-side argument must be a length-3 vector.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.Transform() + 1

    match = (
        "Unsupported operand type(s) for +: 'Transform' and 'dict'\n"
        'The right-side argument must be a length-3 vector.'
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
        'The right-side argument must be a single number or a length-3 vector '
        'or have 3x3 or 4x4 shape.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.Transform() * (1, 2, 3, 4)

    match = (
        "Unsupported operand type(s) for *: 'Transform' and 'dict'\n"
        'The right-side argument must be transform-like.'
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        pv.Transform() * {}


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
    transform.compose(pv.transformations.axis_angle_rotation((0, 0, 1), 45))
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


def test_transform_shear_matrix(transform):
    assert not transform.has_shear
    transform.compose(SHEAR)
    assert transform.has_shear
    assert np.allclose(transform.shear_matrix, SHEAR)


@pytest.mark.parametrize('do_shear', [True, False])
@pytest.mark.parametrize('do_scale', [True, False])
@pytest.mark.parametrize('do_reflection', [True, False])
@pytest.mark.parametrize('do_rotate', [True, False])
@pytest.mark.parametrize('do_translate', [True, False])
def test_transform_decompose(
    transform, do_shear, do_scale, do_reflection, do_rotate, do_translate
):
    if do_shear:
        transform.compose(SHEAR)
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


@pytest.mark.parametrize(
    ('representation', 'args', 'expected_type', 'expected_shape'),
    [
        (None, (), Rotation, None),
        ('quat', (), np.ndarray, (4,)),
        ('matrix', (), np.ndarray, (3, 3)),
        ('rotvec', (), np.ndarray, (3,)),
        ('mrp', (), np.ndarray, (3,)),
        ('euler', ('xyz',), np.ndarray, (3,)),
        ('davenport', (np.eye(3), 'extrinsic'), np.ndarray, (3,)),
    ],
)
def test_transform_as_rotation(representation, args, expected_type, expected_shape):
    out = pv.Transform().as_rotation(representation, *args)
    assert isinstance(out, expected_type)
    if expected_shape:
        assert out.shape == expected_shape


@pytest.mark.parametrize(
    ('event', 'expected'),
    [
        ('end', _vtk.vtkCommand.EndInteractionEvent),
        ('start', _vtk.vtkCommand.StartInteractionEvent),
        ('always', _vtk.vtkCommand.InteractionEvent),
        (_vtk.vtkCommand.InteractionEvent,) * 2,
        (_vtk.vtkCommand.EndInteractionEvent,) * 2,
        (_vtk.vtkCommand.StartInteractionEvent,) * 2,
    ],
)
def test_parse_interaction_event(
    event: str | _vtk.vtkCommand.EventIds,
    expected: _vtk.vtkCommand.EventIds,
):
    assert _parse_interaction_event(event) == expected


def test_parse_interaction_event_raises_str():
    with pytest.raises(
        ValueError,
        match=r'Expected.*start.*end.*always.*foo was given',
    ):
        _parse_interaction_event('foo')


def test_parse_interaction_event_raises_wrong_type():
    with pytest.raises(
        TypeError,
        match=r'.*either a str or.*vtk.vtkCommand.EventIds.*int.* was given',
    ):
        _parse_interaction_event(1)


def test_classproperty():
    magic_number = 42

    class Foo:
        @_classproperty
        def prop(cls):  # noqa: N805
            return magic_number

    assert Foo.prop == magic_number
    assert Foo().prop == magic_number

    with pytest.raises(TypeError, match='object is not callable'):
        Foo.prop()
    with pytest.raises(TypeError, match='object is not callable'):
        Foo().prop()


@pytest.fixture
def modifies_verbosity():
    initial_verbosity = _vtk.vtkLogger.GetCurrentVerbosityCutoff()
    yield
    _vtk.vtkLogger.SetStderrVerbosity(initial_verbosity)


@pytest.mark.usefixtures('modifies_verbosity')
@pytest.mark.parametrize(
    'verbosity',
    [
        'off',
        'error',
        'warning',
        'info',
        'max',
    ],
)
def test_vtk_verbosity_context(verbosity):
    initial_verbosity = _vtk.vtkLogger.VERBOSITY_OFF
    _vtk.vtkLogger.SetStderrVerbosity(initial_verbosity)
    with pv.vtk_verbosity(verbosity):
        ...
    assert _vtk.vtkLogger.GetCurrentVerbosityCutoff() == initial_verbosity


@pytest.mark.usefixtures('modifies_verbosity')
def test_vtk_verbosity_nested_context():
    LEVEL1 = 'off'
    LEVEL2 = 'error'
    LEVEL3 = 'warning'
    with pv.vtk_verbosity(LEVEL1):
        with pv.vtk_verbosity(LEVEL2):
            with pv.vtk_verbosity(LEVEL3):
                assert pv.vtk_verbosity() == LEVEL3
            assert pv.vtk_verbosity() == LEVEL2
        assert pv.vtk_verbosity() == LEVEL1


@pytest.mark.usefixtures('modifies_verbosity')
def test_vtk_verbosity_no_context():
    match = re.escape('State must be set before using it as a context manager.')
    with pytest.raises(ValueError, match=match):
        with pv.vtk_verbosity:
            ...

    # Use context normally
    with pv.vtk_verbosity('off'):
        ...

    # Test again to check reset after use
    with pytest.raises(ValueError, match=match):
        with pv.vtk_verbosity:
            ...


@pytest.mark.usefixtures('modifies_verbosity')
def test_vtk_verbosity_set_get():
    assert _vtk.vtkLogger.GetCurrentVerbosityCutoff() != _vtk.vtkLogger.VERBOSITY_OFF
    pv.vtk_verbosity('off')
    assert pv.vtk_verbosity() == 'off'
    assert _vtk.vtkLogger.GetCurrentVerbosityCutoff() == _vtk.vtkLogger.VERBOSITY_OFF

    # Set this to an invalid state with vtk methods
    _vtk.vtkLogger.SetStderrVerbosity(_vtk.vtkLogger.VERBOSITY_1)
    with pytest.raises(ValueError, match="state '1' is not valid"):
        pv.vtk_verbosity()


@pytest.mark.parametrize('value', ['str', 'invalid'])
def test_vtk_verbosity_invalid_input(value):
    match = re.escape("state must be one of: \n\t('off', 'error', 'warning', 'info', 'max')")
    with pytest.raises(ValueError, match=match):
        with pv.vtk_verbosity(value):
            ...


@pytest.mark.needs_vtk_version(9, 4)
def test_vtk_snake_case():
    assert pv.vtk_snake_case() == 'error'
    match = "The attribute 'information' is defined by VTK and is not part of the PyVista API"

    with pytest.raises(pv.PyVistaAttributeError, match=match):
        _ = pv.PolyData().information

    pv.vtk_snake_case('allow')
    assert pv.vtk_snake_case() == 'allow'
    _ = pv.PolyData().information

    with pv.vtk_snake_case('warning'):
        with pytest.warns(RuntimeWarning, match=match):
            _ = pv.PolyData().information


def test_allow_new_attributes():
    match = (
        "Attribute '_?foo' does not exist and cannot be added to class 'PolyData'\nUse "
        '`pyvista.set_new_attribute` or `pyvista.allow_new_attributes` to set new attributes.'
    )

    def set_private():
        _ = pv.PolyData()._foo = 42

    def set_public():
        _ = pv.PolyData().foo = 42

    pv.allow_new_attributes(False)
    assert pv.allow_new_attributes() is False
    with pytest.raises(pv.PyVistaAttributeError, match=match):
        set_private()
    with pytest.raises(pv.PyVistaAttributeError, match=match):
        set_public()

    pv.allow_new_attributes('private')
    assert pv.allow_new_attributes() == 'private'
    set_private()
    with pytest.raises(pv.PyVistaAttributeError, match=match):
        set_public()

    pv.allow_new_attributes(True)
    assert pv.allow_new_attributes() is True
    set_private()
    set_public()


T = TypeVar('T')


def _create_state_manager_subclass(arg1, arg2=None, sub_subclass=False):
    if arg2 is not None:

        class MyState(_StateManager[arg1, arg2]):
            @property
            def _state(self): ...

            @_state.setter
            def _state(self, state): ...
    else:

        class MyState(_StateManager[arg1]):
            @property
            def _state(self): ...

            @_state.setter
            def _state(self, state): ...

    if sub_subclass:

        class MyState2(MyState): ...

        return MyState2
    return MyState


@pytest.mark.parametrize('arg', [T, int, Literal, TypeVar, [int, float], [[int], [float]]])
def test_state_manager_invalid_type_arg(arg):
    if isinstance(arg, Iterable):
        cls = _create_state_manager_subclass(*arg)
    else:
        cls = _create_state_manager_subclass(arg)

    match = (
        'Type argument for subclasses must be a single non-empty Literal with all '
        'state options provided.'
    )
    with pytest.raises(TypeError, match=match):
        cls()


def test_state_manager_sub_subclass():
    options = Literal['on', 'off']
    cls = _create_state_manager_subclass(options, sub_subclass=True)
    manager = cls()
    manager('on')
    assert manager._valid_states == get_args(options)


@pytest.mark.parametrize(
    'cell_type',
    [pv.CellType.TRIANGLE, int(pv.CellType.TRIANGLE), 'triangle', 'TRIANGLE'],
)
def test_cell_quality_info(cell_type):
    measure = 'area'
    info = pv.cell_quality_info(cell_type, measure)
    assert isinstance(info, CellQualityInfo)
    assert info.cell_type == pv.CellType.TRIANGLE
    assert info.quality_measure == measure


CELL_QUALITY_IDS = [f'{info.cell_type.name}-{info.quality_measure}' for info in _CELL_QUALITY_INFO]


def _compute_unit_cell_quality(
    info: CellQualityInfo,
    null_value=-42.42,
    coincident: Literal['all', 'single', False] = False,
):
    example_name = _CELL_TYPE_INFO[info.cell_type.name].example
    cell_mesh = getattr(ex.cells, example_name)()
    if coincident == 'all':
        cell_mesh.points[:] = 0.0
    elif coincident == 'single':
        cell_mesh.points[1] = cell_mesh.points[0]
    qual = cell_mesh.cell_quality(info.quality_measure, null_value=null_value)
    return qual.active_scalars[0]


@parametrize('info', _CELL_QUALITY_INFO, ids=CELL_QUALITY_IDS)
def test_cell_quality_info_valid_measures(info):
    # Ensure the computed measure is not null
    null_value = -1
    qual_value = _compute_unit_cell_quality(info, null_value)
    if np.isclose(qual_value, null_value):
        pytest.fail(
            f'Measure {info.quality_measure!r} is not valid for cell type {info.cell_type.name!r}'
        )


def xfail_wedge_negative_volume(info):
    if info.cell_type == pv.CellType.WEDGE and info.quality_measure == 'volume':
        pytest.xfail(
            'vtkWedge returns negative volume, see https://gitlab.kitware.com/vtk/vtk/-/issues/19643'
        )


def xfail_distortion_returns_one(info):
    if (
        info.cell_type in [pv.CellType.TRIANGLE, pv.CellType.TETRA]
        and info.quality_measure == 'distortion'
    ):
        pytest.xfail(
            'Distortion always returns one, see https://gitlab.kitware.com/vtk/vtk/-/issues/19646.'
        )


@parametrize('info', _CELL_QUALITY_INFO, ids=CELL_QUALITY_IDS)
def test_cell_quality_info_unit_cell_value(info):
    """Test that the actual computed measure for a unit cell matches the reported value."""
    xfail_wedge_negative_volume(info)

    unit_cell_value = info.unit_cell_value
    qual_value = _compute_unit_cell_quality(info)
    assert np.isclose(qual_value, unit_cell_value)


@parametrize('info', _CELL_QUALITY_INFO, ids=CELL_QUALITY_IDS)
def test_cell_quality_info_acceptable_range(info):
    """Test that the unit cell value is within the acceptable range."""
    # Some cells / measures have bugs and return invalid values and are expected to fail
    xfail_wedge_negative_volume(info)

    acceptable_range = info.acceptable_range
    unit_cell_value = info.unit_cell_value

    assert unit_cell_value >= acceptable_range[0]
    assert unit_cell_value <= acceptable_range[1]


def _replace_range_infinity(rng):
    rng = list(rng)
    lower, upper = rng
    if lower == -float('inf'):
        rng[0] = np.finfo(lower).min
    if upper == float('inf'):
        rng[1] = np.finfo(upper).max
    return rng


@parametrize('info', _CELL_QUALITY_INFO, ids=CELL_QUALITY_IDS)
def test_cell_quality_info_normal_range(info):
    """Test that the normal range is broader than the acceptable range."""
    acceptable_range = _replace_range_infinity(info.acceptable_range)
    normal_range = _replace_range_infinity(info.normal_range)

    assert normal_range[0] <= acceptable_range[0]
    assert normal_range[1] >= acceptable_range[1]


@parametrize('info', _CELL_QUALITY_INFO, ids=CELL_QUALITY_IDS)
def test_cell_quality_info_full_range(info):
    """Test that the full range is broader than the normal range."""
    normal_range = _replace_range_infinity(info.normal_range)
    full_range = _replace_range_infinity(info.full_range)

    assert full_range[0] <= normal_range[0]
    assert full_range[1] >= normal_range[1]


@parametrize('info', _CELL_QUALITY_INFO, ids=CELL_QUALITY_IDS)
def test_cell_quality_info_degenerate_cell(info):
    # Some cells / measures have bugs and return invalid values and are expected to fail
    xfail_distortion_returns_one(info)

    # Compare non-generate cell with degenerate cell with coincident point(s)
    unit_cell_quality = _compute_unit_cell_quality(info, coincident=False)
    all_coincident_quality = _compute_unit_cell_quality(info, coincident='all')
    single_coincident_quality = _compute_unit_cell_quality(info, coincident='single')

    # Quality must differ in at least one of the degeneracy cases
    assert (not np.isclose(unit_cell_quality, all_coincident_quality)) or (
        not np.isclose(unit_cell_quality, single_coincident_quality)
    )


def test_cell_quality_info_raises():
    match = re.escape(
        "Cell quality info is not available for cell type 'QUADRATIC_EDGE'. Valid options are:\n"
        "['TRIANGLE', 'QUAD', 'TETRA', 'HEXAHEDRON', 'PYRAMID', 'WEDGE']"
    )
    with pytest.raises(ValueError, match=match):
        pv.cell_quality_info(pv.CellType.QUADRATIC_EDGE, 'area')
    with pytest.raises(ValueError, match=match):
        pv.cell_quality_info(pv.CellType.QUADRATIC_EDGE.name, 'area')

    match = re.escape(
        "Cell quality info is not available for 'TRIANGLE' measure 'volume'. Valid options are:\n"
        "['area', 'aspect_ratio', 'aspect_frobenius', 'condition', 'distortion', "
        "'max_angle', 'min_angle', 'scaled_jacobian', 'radius_ratio', 'shape', 'shape_and_size']"
    )
    with pytest.raises(ValueError, match=match):
        pv.cell_quality_info(pv.CellType.TRIANGLE, 'volume')


@pytest.mark.needs_vtk_version(9, 4)
def test_is_vtk_attribute():
    assert is_vtk_attribute(pv.ImageData(), 'GetCells')
    assert is_vtk_attribute(pv.UnstructuredGrid(), 'GetCells')

    assert is_vtk_attribute(pv.ImageData(), 'cells')
    assert not is_vtk_attribute(pv.UnstructuredGrid(), 'cells')

    assert not is_vtk_attribute(pv.ImageData, 'foo')


@pytest.mark.parametrize('obj', [pv.ImageData(), pv.ImageData])
@pytest.mark.needs_vtk_version(9, 4)
def test_is_vtk_attribute_input_type(obj):
    assert is_vtk_attribute(obj, 'GetDimensions')


warnings.simplefilter('always')


def test_deprecate_positional_args_error_messages():
    # Test single arg
    @_deprecate_positional_args
    def foo(bar): ...

    match = (
        "Argument 'bar' must be passed as a keyword argument to function "
        "'test_deprecate_positional_args_error_messages.<locals>.foo'.\n"
        'From version 0.50, passing this as a positional argument will result in a TypeError.'
    )
    with pytest.warns(pv.PyVistaDeprecationWarning, match=match):
        foo(True)

    # Test many args
    @_deprecate_positional_args(version=(1, 2))
    def foo(bar, baz): ...

    match = (
        "Arguments 'bar', 'baz' must be passed as keyword arguments to function "
        "'test_deprecate_positional_args_error_messages.<locals>.foo'.\n"
        'From version 1.2, passing these as positional arguments will result in a TypeError.'
    )
    with pytest.warns(pv.PyVistaDeprecationWarning, match=match):
        foo(True, True)


def test_deprecate_positional_args_post_deprecation():
    match = (
        r'Positional arguments are no longer allowed in '
        r"'test_deprecate_positional_args_post_deprecation.<locals>.foo'\.\n"
        r'Update the function signature at:\n'
        r'.*test_utilities\.py:\d+ to enforce keyword-only args:\n'
        r'    test_deprecate_positional_args_post_deprecation.<locals>.foo\(bar, \*, baz\)\n'
        r"and remove the '_deprecate_positional_args' decorator\."
    )
    with pytest.raises(RuntimeError, match=match):

        @_deprecate_positional_args(allowed=['bar'], version=(0, 46))
        def foo(bar, baz): ...

    match = 'foo(*, bar, baz, ham, ...)'
    with pytest.raises(RuntimeError, match=re.escape(match)):

        @_deprecate_positional_args(version=(0, 46))
        def foo(bar, baz, ham, eggs): ...

    match = 'foo(self, *, bar, baz, ham, ...)'
    with pytest.raises(RuntimeError, match=re.escape(match)):

        class Foo:
            @_deprecate_positional_args(version=(0, 46))
            def foo(self, bar, baz, ham, eggs): ...


def test_deprecate_positional_args_allowed():
    # Test single allowed
    @_deprecate_positional_args(allowed=['bar'])
    def foo(bar, baz): ...

    foo(True, baz=True)

    # Too many allowed args
    match = (
        "In decorator '_deprecate_positional_args' for function "
        "'test_deprecate_positional_args_allowed.<locals>.foo':\n"
        f'A maximum of {_MAX_POSITIONAL_ARGS} positional arguments are allowed.\n'
        "Got 6: ['bar', 'baz', 'qux', 'ham', 'eggs', 'cats']"
    )
    with pytest.raises(ValueError, match=re.escape(match)):

        @_deprecate_positional_args(allowed=['bar', 'baz', 'qux', 'ham', 'eggs', 'cats'])
        def foo(bar, baz, qux, ham, eggs, cats): ...

    # Test invalid allowed
    match = (
        "Allowed positional argument 'invalid' in decorator '_deprecate_positional_args'\n"
        'is not a parameter of '
        "function 'test_deprecate_positional_args_allowed.<locals>.foo'."
    )
    with pytest.raises(ValueError, match=re.escape(match)):

        @_deprecate_positional_args(allowed=['invalid'])
        def foo(bar): ...

    match = (
        "In decorator '_deprecate_positional_args' for function "
        "'test_deprecate_positional_args_allowed.<locals>.foo':\n"
        "Allowed arguments must be a list, got <class 'str'>."
    )
    with pytest.raises(TypeError, match=re.escape(match)):

        @_deprecate_positional_args(allowed='invalid')
        def foo(bar): ...

    # Test invalid order
    match = (
        "The `allowed` list ['b', 'a'] in decorator '_deprecate_positional_args' "
        'is not in the\nsame order as the parameters in '
        "'test_deprecate_positional_args_allowed.<locals>.foo'.\n"
        "Expected order: ['a', 'b']."
    )
    with pytest.raises(ValueError, match=re.escape(match)):

        @_deprecate_positional_args(allowed=['b', 'a'])
        def foo(a, b, c): ...

    # Test not already kwonly
    match = (
        "Parameter 'b' in decorator '_deprecate_positional_args' is already keyword-only\n"
        'and should be removed from the allowed list.'
    )
    with pytest.raises(ValueError, match=match):

        @_deprecate_positional_args(allowed=['a', 'b'])
        def foo(a, *, b): ...


def test_deprecate_positional_args_n_allowed():
    n_allowed = 4
    assert n_allowed > _MAX_POSITIONAL_ARGS

    @_deprecate_positional_args(allowed=['a', 'b', 'c', 'd'], n_allowed=4)
    def foo(a, b, c, d, e=True): ...

    match = (
        "In decorator '_deprecate_positional_args' for function "
        "'test_deprecate_positional_args_n_allowed.<locals>.foo':\n"
        '`n_allowed` must be greater than 3 for it to be useful.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):

        @_deprecate_positional_args(allowed=['a', 'b', 'c'], n_allowed=_MAX_POSITIONAL_ARGS)
        def foo(a, b, c): ...


def test_deprecate_positional_args_class_methods():
    # Test that 'cls' and 'self' args do not cause problems
    class Foo:
        @classmethod
        @_deprecate_positional_args
        def foo_classmethod(cls, bar=None): ...

        @_deprecate_positional_args
        def foo_method(self, bar=None): ...

    obj = Foo()
    obj.foo_method()
    obj.foo_classmethod()


def test_deprecate_positional_args_decorator_not_needed():
    match = (
        "Function 'test_deprecate_positional_args_decorator_not_needed.<locals>.Foo.foo' has 0 "
        'positional arguments, which is less than or equal to the\nmaximum number of allowed '
        f'positional arguments ({_MAX_POSITIONAL_ARGS}).\n'
        f'This decorator is not necessary and can be removed.'
    )
    with pytest.raises(RuntimeError, match=re.escape(match)):

        class Foo:
            @classmethod
            @_deprecate_positional_args
            def foo(cls, *, bar=None): ...

    with pytest.raises(RuntimeError, match=re.escape(match)):

        class Foo:
            @_deprecate_positional_args
            def foo(self, *, bar=None): ...

    match = (
        f"Function 'test_deprecate_positional_args_decorator_not_needed.<locals>.foo' has 3 "
        f'positional arguments, which is less than or equal to the\nmaximum number of allowed '
        f'positional arguments ({_MAX_POSITIONAL_ARGS}).\n'
        f'This decorator is not necessary and can be removed.'
    )
    with pytest.raises(RuntimeError, match=re.escape(match)):

        @_deprecate_positional_args(allowed=['a', 'b', 'c'])
        def foo(a, b, c): ...


@pytest.mark.skipif(
    sys.version_info < (3, 11) or sys.platform == 'darwin',
    reason='Requires Python 3.11+, path issues on macOS',
)
def test_max_positional_args_matches_pyproject():
    root = Path(
        os.environ.get('TOX_ROOT', Path(pv.__file__).parents[1])
    )  # to make the test work when pyvista is installed via tox
    pyproject_path = root / 'pyproject.toml'
    with pyproject_path.open('rb') as f:
        pyproject_data = tomllib.load(f)
    expected_value = pyproject_data['tool']['ruff']['lint']['pylint']['max-positional-args']

    assert expected_value == _MAX_POSITIONAL_ARGS


@pytest.mark.parametrize('compression', get_args(_CompressionOptions))
def test_writer_compression(compression):
    writer = pv.XMLUnstructuredGridWriter('', pv.UnstructuredGrid())
    writer.compression = compression
    if compression is None:
        assert writer.writer.GetCompressor() is None
    else:
        compressor = writer.writer.GetCompressor()
        assert compression in str(type(compressor)).lower()


def get_concrete_classes(module, abc):
    """Collect all concrete BaseWriter subclasses"""
    concrete_classes = []
    for obj in vars(module).values():
        if not (inspect.isclass(obj) and issubclass(obj, abc)):
            continue
        # Skip abstract
        try:
            obj()
        except TypeError as e:
            if 'abstract' in repr(e):
                continue
        concrete_classes.append(obj)
    return concrete_classes


WRITER_CLASSES = get_concrete_classes(pv.core.utilities.writer, pv.BaseWriter)
READER_CLASSES = get_concrete_classes(pv.core.utilities.reader, pv.BaseReader)


@pytest.mark.parametrize('writer_cls', WRITER_CLASSES)
def test_writer_data_mode_mixin(writer_cls):
    """Test that classes with an ascii setter have a data_mode property."""
    if writer_cls is pv.HDFWriter and pv.vtk_version_info < (9, 4, 0):
        pytest.xfail('Needs vtk 9.4')
    if not any('ascii' in attr.lower() for attr in dir(writer_cls._vtk_class)):
        pytest.skip(f'{writer_cls.__name__} does not support ASCII mode, skipping')

    assert _DataFormatMixin in writer_cls.__mro__, f'{writer_cls.__name__} missing DataModeMixin'
    mesh = (
        pv.PartitionedDataSet() if writer_cls is pv.XMLPartitionedDataSetWriter else pv.PolyData()
    )

    obj = writer_cls('', mesh)
    assert obj.data_format == 'binary'
    obj.data_format = 'ascii'
    assert obj.data_format == 'ascii'
    obj.data_format = 'binary'
    assert obj.data_format == 'binary'


@pytest.mark.parametrize('cls', [*READER_CLASSES, *WRITER_CLASSES])
def test_fileio_extensions(cls):
    if cls is pv.HDFWriter and pv.vtk_version_info < (9, 4, 0):
        pytest.xfail('Needs vtk 9.4')
    if cls in [pv.OpenFOAMReader, pv.MultiBlockPlot3DReader]:
        # These classes are not associated with any extensions
        pytest.xfail()
    assert len(cls.extensions) > 0


def test_ply_writer(sphere, tmp_path):
    writer_cls = pv.PLYWriter
    assert writer_cls.extensions == ('.ply',)

    path = tmp_path / 'sphere.ply'
    writer = writer_cls(path, sphere)
    assert writer.path == str(path)

    if not sys.platform.startswith('win'):
        # Skip repr check on Windows due to escaped backslashes
        assert repr(writer) == f'PLYWriter({str(path)!r})'

    array = np.arange(sphere.n_points)
    with pytest.raises(TypeError, match='incorrect dtype'):
        writer.texture = array
    array = array.astype('uint8')

    # Test array is implicitly added to mesh
    texture_name = '_color_array'
    writer.texture = array
    assert writer.texture == texture_name
    writer.texture = texture_name
    assert writer.texture == texture_name
