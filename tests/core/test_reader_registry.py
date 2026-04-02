"""Tests for custom reader registration and discovery."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import pyvista as pv
from pyvista.core.utilities import reader_registry as _reg_mod
from pyvista.core.utilities.reader_registry import _custom_ext_cloud
from pyvista.core.utilities.reader_registry import _custom_ext_readers
from pyvista.core.utilities.reader_registry import _custom_scheme_readers
from pyvista.core.utilities.reader_registry import _ext_supports_cloud
from pyvista.core.utilities.reader_registry import _get_ext_handler
from pyvista.core.utilities.reader_registry import _get_scheme_handler
from pyvista.core.utilities.reader_registry import _has_scheme
from pyvista.core.utilities.reader_registry import _restore_registry_state
from pyvista.core.utilities.reader_registry import _save_registry_state
from pyvista.core.utilities.reader_registry import _temp_files
from pyvista.core.utilities.reader_registry import register_reader


@pytest.fixture(autouse=True)
def _clean_custom_readers():
    """Remove any custom readers registered during tests."""
    state = _save_registry_state()
    orig_loaded = _reg_mod._entry_points_loaded
    yield
    _restore_registry_state(state)
    _reg_mod._entry_points_loaded = orig_loaded


def _mock_reader(_path, **_kwargs):
    return pv.PolyData()


def test_register_extension():
    register_reader('.myformat', _mock_reader)
    assert '.myformat' in _custom_ext_readers


def test_register_extension_without_dot():
    register_reader('myformat', _mock_reader)
    assert '.myformat' in _custom_ext_readers


def test_register_extension_case_insensitive():
    register_reader('.MyFormat', _mock_reader)
    assert '.myformat' in _custom_ext_readers


def test_register_scheme():
    register_reader('s3://', _mock_reader)
    assert 's3://' in _custom_scheme_readers


def test_register_builtin_collision():
    with pytest.raises(ValueError, match='collides with built-in VTK reader'):
        register_reader('.vtk', _mock_reader)


def test_register_builtin_override():
    register_reader('.vtk', _mock_reader, override=True)
    assert '.vtk' in _custom_ext_readers


def test_register_as_decorator():
    @register_reader('.decorated')
    def my_reader(_path, **_kwargs):
        return pv.PolyData()

    assert '.decorated' in _custom_ext_readers
    assert _custom_ext_readers['.decorated'] is my_reader


def test_register_decorator_stacked():
    @register_reader('.ext1')
    @register_reader('.ext2')
    def my_reader(_path, **_kwargs):
        return pv.PolyData()

    assert _custom_ext_readers['.ext1'] is my_reader
    assert _custom_ext_readers['.ext2'] is my_reader


def test_register_decorator_scheme():
    @register_reader('custom://')
    def my_reader(_path, **_kwargs):
        return pv.PolyData()

    assert _custom_scheme_readers['custom://'] is my_reader


def test_get_ext_handler():
    register_reader('.myformat', _mock_reader)
    assert _get_ext_handler('.myformat') is _mock_reader


def test_get_ext_handler_returns_none():
    assert _get_ext_handler('.nonexistent') is None


def test_get_scheme_handler():
    register_reader('s3://', _mock_reader)
    result = _get_scheme_handler('s3://bucket/key.vtp')
    assert result is not None
    handler, scheme = result
    assert handler is _mock_reader
    assert scheme == 's3://'


def test_get_scheme_handler_case_insensitive():
    register_reader('S3://', _mock_reader)
    result = _get_scheme_handler('s3://bucket/key.vtp')
    assert result is not None


def test_get_scheme_handler_returns_none():
    assert _get_scheme_handler('/local/path.vtp') is None


def test_entry_point_discovery_extension():
    _reg_mod._entry_points_loaded = False

    mock_ep = MagicMock()
    mock_ep.name = '.discovered'
    mock_ep.load.return_value = _mock_reader

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        handler = _get_ext_handler('.discovered')
        assert handler is _mock_reader


def test_entry_point_discovery_scheme():
    _reg_mod._entry_points_loaded = False

    mock_ep = MagicMock()
    mock_ep.name = 'gs://'
    mock_ep.load.return_value = _mock_reader

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        result = _get_scheme_handler('gs://bucket/file.vtp')
        assert result is not None
        handler, _scheme = result
        assert handler is _mock_reader


def test_entry_point_does_not_override_explicit():
    register_reader('.myext', _mock_reader)
    _reg_mod._entry_points_loaded = False

    other_handler = MagicMock(return_value=pv.PolyData())
    mock_ep = MagicMock()
    mock_ep.name = '.myext'
    mock_ep.load.return_value = other_handler

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        handler = _get_ext_handler('.myext')
        assert handler is _mock_reader  # original registration wins


def test_read_with_custom_scheme():
    mock = MagicMock(return_value=pv.PolyData())
    register_reader('custom://', mock)
    result = pv.read('custom://some/resource')
    mock.assert_called_once_with('custom://some/resource')
    assert isinstance(result, pv.PolyData)


def test_read_with_custom_extension(tmp_path):
    test_file = tmp_path / 'data.myext'
    test_file.touch()
    mock = MagicMock(return_value=pv.PolyData())
    register_reader('.myext', mock)
    result = pv.read(str(test_file))
    mock.assert_called_once_with(str(test_file.resolve()))
    assert isinstance(result, pv.PolyData)


def test_cloud_handler_receives_uri():
    """cloud=True handler gets the raw URI, no download."""
    mock = MagicMock(return_value=pv.PolyData())
    register_reader('.mycloud', mock, cloud=True)
    result = pv.read('https://example.com/data.mycloud')
    mock.assert_called_once_with('https://example.com/data.mycloud')
    assert isinstance(result, pv.PolyData)


def test_non_cloud_handler_gets_downloaded_file(tmp_path):
    """cloud=False handler gets a local temp path for http URIs."""
    mock = MagicMock(return_value=pv.PolyData())
    register_reader('.mylocal', mock)

    fake_file = tmp_path / 'downloaded.mylocal'
    fake_file.touch()

    with patch.dict('sys.modules', {'fsspec': None}):
        with patch('pooch.retrieve', return_value=str(fake_file)):
            pv.read('https://example.com/data.mylocal')

    mock.assert_called_once()
    called_path = mock.call_args[0][0]
    assert not called_path.startswith('http')


def test_s3_without_fsspec_raises():
    """s3:// URI without fsspec raises ImportError with install hint."""
    register_reader('.myext', _mock_reader)
    with patch.dict('sys.modules', {'fsspec': None}):
        with pytest.raises(ImportError, match='fsspec'):
            pv.read('s3://bucket/data.myext')


def test_http_download_builtin_extension(tmp_path):
    """http URI with a built-in extension downloads then reads locally."""
    mesh = pv.Sphere()
    vtp_file = tmp_path / 'mesh.vtp'
    mesh.save(vtp_file)

    with patch.dict('sys.modules', {'fsspec': None}):
        with patch('pooch.retrieve', return_value=str(vtp_file)) as dl:
            result = pv.read('https://example.com/mesh.vtp')

    dl.assert_called_once()
    assert isinstance(result, pv.PolyData)
    assert result.n_points == mesh.n_points


def test_cloud_flag_stored():
    """cloud flag is accessible via _ext_supports_cloud."""
    register_reader('.native', _mock_reader, cloud=True)
    register_reader('.local', _mock_reader, cloud=False)
    assert _ext_supports_cloud('.native') is True
    assert _ext_supports_cloud('.local') is False
    assert _ext_supports_cloud('.unknown') is False


def test_decorator_with_cloud_true():
    """Decorator form with cloud=True stores the flag."""

    @register_reader('.cloudext', cloud=True)
    def my_reader(_path, **_kwargs):
        return pv.PolyData()

    assert _custom_ext_cloud['.cloudext'] is True
    assert _custom_ext_readers['.cloudext'] is my_reader


def test_has_scheme_rejects_local_paths():
    """Paths with :// after a slash are not URIs."""
    assert _has_scheme('https://example.com/mesh.vtp') is True
    assert _has_scheme('s3://bucket/key') is True
    assert _has_scheme('/data/re://fresh/mesh.vtu') is False
    assert _has_scheme('mesh.vtu') is False
    assert _has_scheme('') is False


def test_temp_file_cleanup(tmp_path):
    """Downloaded temp files are tracked for cleanup."""
    initial = len(_temp_files)
    mock = MagicMock(return_value=pv.PolyData())
    register_reader('.tmptest', mock)

    fake_file = tmp_path / 'fake.tmptest'
    fake_file.touch()

    with patch.dict('sys.modules', {'fsspec': None}):
        with patch('pooch.retrieve', return_value=str(fake_file)):
            pv.read('https://example.com/data.tmptest')

    assert len(_temp_files) > initial


def test_top_level_export():
    assert hasattr(pv, 'register_reader')
    assert pv.register_reader is register_reader
