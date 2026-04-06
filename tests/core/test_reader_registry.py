"""Tests for custom reader registration and discovery."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import re
import types
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

import pytest

import pyvista as pv
import pyvista.core.utilities as pv_utilities
from pyvista.core.utilities import fileio as _fileio_mod
from pyvista.core.utilities import reader_registry as _reg_mod


@pytest.fixture(autouse=True)
def _clean_custom_readers():
    """Remove any custom readers registered during tests."""
    state = _reg_mod._save_registry_state()
    orig_loaded = _reg_mod._entry_points_loaded
    orig_temp_files = list(_reg_mod._temp_files)
    yield
    _reg_mod._temp_files[:] = [
        path for path in _reg_mod._temp_files if path not in orig_temp_files
    ]
    _reg_mod._cleanup_temp_files()
    _reg_mod._temp_files.extend(orig_temp_files)
    _reg_mod._restore_registry_state(state)
    _reg_mod._entry_points_loaded = orig_loaded


def _mock_reader(_path, **__):
    return pv.PolyData()


def test_module_import_registers_cleanup_handler():
    module_path = Path(_reg_mod.__file__)
    spec = importlib.util.spec_from_file_location('reader_registry_test_copy', module_path)
    assert spec is not None
    assert spec.loader is not None

    with patch('atexit.register') as register:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    register.assert_called_once_with(module._cleanup_temp_files)
    assert module._custom_ext_readers == {}
    assert module._entry_points_loaded is False
    assert module._temp_files == []


def test_register_extension():
    pv.register_reader('.myformat', _mock_reader)
    assert '.myformat' in _reg_mod._custom_ext_readers


def test_register_extension_without_dot():
    pv.register_reader('myformat', _mock_reader)
    assert '.myformat' in _reg_mod._custom_ext_readers


def test_register_extension_case_insensitive():
    pv.register_reader('.MyFormat', _mock_reader)
    assert '.myformat' in _reg_mod._custom_ext_readers


def test_register_builtin_collision():
    with pytest.raises(ValueError, match='collides with built-in VTK reader'):
        pv.register_reader('.vtk', _mock_reader)


def test_register_builtin_override():
    pv.register_reader('.vtk', _mock_reader, override=True)
    assert '.vtk' in _reg_mod._custom_ext_readers


def test_register_as_decorator():
    @pv.register_reader('.decorated')
    def my_reader(_path, **__):
        return pv.PolyData()

    assert '.decorated' in _reg_mod._custom_ext_readers
    assert _reg_mod._custom_ext_readers['.decorated'] is my_reader


def test_register_decorator_stacked():
    @pv.register_reader('.ext1')
    @pv.register_reader('.ext2')
    def my_reader(_path, **__):
        return pv.PolyData()

    assert _reg_mod._custom_ext_readers['.ext1'] is my_reader
    assert _reg_mod._custom_ext_readers['.ext2'] is my_reader


def test_get_ext_handler():
    pv.register_reader('.myformat', _mock_reader)
    assert _reg_mod._get_ext_handler('.myformat') is _mock_reader


def test_get_ext_handler_returns_none():
    assert _reg_mod._get_ext_handler('.nonexistent') is None


def test_entry_point_discovery_extension():
    _reg_mod._entry_points_loaded = False

    mock_ep = MagicMock()
    mock_ep.name = '.discovered'
    mock_ep.load.return_value = _mock_reader

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        handler = _reg_mod._get_ext_handler('.discovered')
        assert handler is _mock_reader


def test_entry_point_duplicate_warns_and_uses_first_handler():
    _reg_mod._entry_points_loaded = False

    first = MagicMock()
    first.name = 'discovered'
    first.value = 'package:first'
    first.load.return_value = _mock_reader

    second = MagicMock()
    second.name = 'discovered'
    second.value = 'package:second'

    with (
        patch(
            'pyvista.core.utilities.reader_registry.entry_points',
            return_value=[first, second],
        ),
        patch('pyvista.core.utilities.reader_registry.warn_external') as warn,
    ):
        handler = _reg_mod._get_ext_handler('.discovered')

    assert handler is _mock_reader
    second.load.assert_not_called()
    warn.assert_called_once()
    assert warn.call_args.args[0] == (
        'Multiple pyvista.readers entry points registered for ".discovered": '
        'package:first, package:second. Using package:first.'
    )


def test_entry_point_does_not_override_explicit():
    pv.register_reader('.myext', _mock_reader)
    _reg_mod._entry_points_loaded = False

    other_handler = MagicMock(return_value=pv.PolyData())
    mock_ep = MagicMock()
    mock_ep.name = '.myext'
    mock_ep.load.return_value = other_handler

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        handler = _reg_mod._get_ext_handler('.myext')
        assert handler is _mock_reader  # original registration wins


def test_entry_point_load_failure_is_ignored():
    _reg_mod._entry_points_loaded = False

    broken = MagicMock()
    broken.name = '.broken'
    broken.value = 'package:broken'
    broken.load.side_effect = RuntimeError('broken plugin')

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[broken]):
        assert _reg_mod._get_ext_handler('.broken') is None


def test_read_with_custom_extension(tmp_path):
    test_file = tmp_path / 'data.myext'
    test_file.touch()
    mock = MagicMock(return_value=pv.PolyData())
    pv.register_reader('.myext', mock)
    result = pv.read(str(test_file))
    mock.assert_called_once_with(str(test_file.resolve()))
    assert isinstance(result, pv.PolyData)


def test_uri_forwarded_to_custom_reader():
    """Remote URI with a custom extension is passed directly to the handler."""
    mock = MagicMock(return_value=pv.PolyData())
    pv.register_reader('.myformat', mock)

    result = pv.read('https://example.com/data.myformat')
    mock.assert_called_once_with('https://example.com/data.myformat')
    assert isinstance(result, pv.PolyData)


def test_s3_uri_forwarded_to_custom_reader():
    """s3:// URI with a custom extension is passed directly to the handler."""
    mock = MagicMock(return_value=pv.PolyData())
    pv.register_reader('.myformat', mock)

    result = pv.read('s3://bucket/data.myformat')
    mock.assert_called_once_with('s3://bucket/data.myformat')
    assert isinstance(result, pv.PolyData)


def test_uri_fallback_downloads_on_local_file_required(tmp_path):
    """If handler raises LocalFileRequiredError, download and retry."""
    fake_file = tmp_path / 'data.myformat'
    fake_file.touch()

    call_count = 0

    def handler_needs_local(path, **__):
        nonlocal call_count
        call_count += 1
        if pv.has_scheme(path):
            raise pv.LocalFileRequiredError
        return pv.PolyData()

    pv.register_reader('.myformat', handler_needs_local)

    with patch.dict('sys.modules', {'fsspec': None}):
        with patch('pooch.retrieve', return_value=str(fake_file)):
            result = pv.read('https://example.com/data.myformat')

    assert call_count == 2  # first with URI, second with local path
    assert isinstance(result, pv.PolyData)


def test_uri_handler_error_propagates():
    """Non-LocalFileRequiredError exceptions propagate, no retry."""

    def bad_reader(_path, **__):
        msg = 'broken reader'
        raise RuntimeError(msg)

    pv.register_reader('.broken', bad_reader)

    with pytest.raises(RuntimeError, match='broken reader'):
        pv.read('https://example.com/data.broken')


def test_uri_downloads_then_reads_builtin_ext(tmp_path):
    """Remote URI with a built-in extension downloads then reads locally."""
    mesh = pv.Sphere()
    vtp_file = tmp_path / 'mesh.vtp'
    mesh.save(vtp_file)

    with patch.dict('sys.modules', {'fsspec': None}):
        with patch('pooch.retrieve', return_value=str(vtp_file)) as dl:
            result = pv.read('https://example.com/mesh.vtp')

    dl.assert_called_once()
    assert isinstance(result, pv.PolyData)
    assert result.n_points == mesh.n_points


def test_s3_without_fsspec_raises():
    """s3:// URI without fsspec raises ImportError with install hint."""
    with patch.dict('sys.modules', {'fsspec': None}):
        with pytest.raises(ImportError, match='fsspec'):
            pv.read('s3://bucket/data.vtp')


def test_download_uri_uses_fsspec_and_tracks_temp_file():
    payload = b'custom-reader-data'
    fake_fsspec = types.SimpleNamespace(
        open=MagicMock(return_value=contextlib.nullcontext(io.BytesIO(payload))),
    )

    with patch.dict('sys.modules', {'fsspec': fake_fsspec}):
        local_path = _reg_mod._download_uri('s3://bucket/data.myformat', '.myformat')

    assert Path(local_path).suffix == '.myformat'
    assert Path(local_path).read_bytes() == payload
    assert local_path in _reg_mod._temp_files


def test_download_uri_wraps_fsspec_errors():
    fake_fsspec = types.SimpleNamespace(
        open=MagicMock(side_effect=OSError('bucket unavailable')),
    )

    with patch.dict('sys.modules', {'fsspec': fake_fsspec}):
        with pytest.raises(
            ConnectionError,
            match=re.escape(
                'Failed to download "s3://bucket/data.myformat": bucket unavailable',
            ),
        ):
            _reg_mod._download_uri('s3://bucket/data.myformat', '.myformat')

    assert len(_reg_mod._temp_files) == 1
    assert Path(_reg_mod._temp_files[0]).suffix == '.myformat'


def test_cleanup_temp_files_removes_existing_and_missing_paths(tmp_path):
    existing = tmp_path / 'existing.myformat'
    existing.touch()
    missing = tmp_path / 'missing.myformat'
    _reg_mod._temp_files.extend([str(existing), str(missing)])

    _reg_mod._cleanup_temp_files()

    assert not existing.exists()
    assert not missing.exists()
    assert _reg_mod._temp_files == []


def test_has_scheme_rejects_local_paths():
    """Paths with :// after a slash are not URIs."""
    assert pv.has_scheme('https://example.com/mesh.vtp') is True
    assert pv.has_scheme('s3://bucket/key') is True
    assert pv.has_scheme('/data/re://fresh/mesh.vtu') is False
    assert pv.has_scheme('mesh.vtu') is False
    assert pv.has_scheme('') is False


def test_temp_file_cleanup(tmp_path):
    """Downloaded temp files are tracked for cleanup."""
    initial = len(_reg_mod._temp_files)
    mesh = pv.Sphere()
    vtp_file = tmp_path / 'mesh.vtp'
    mesh.save(vtp_file)

    with patch.dict('sys.modules', {'fsspec': None}):
        with patch('pooch.retrieve', return_value=str(vtp_file)):
            pv.read('https://example.com/mesh.vtp')

    assert len(_reg_mod._temp_files) > initial


def test_top_level_exports_survive_reload(tmp_path):
    importlib.reload(_fileio_mod)
    utilities_module = importlib.reload(pv_utilities)
    pyvista_module = importlib.reload(pv)

    assert pyvista_module.register_reader is _reg_mod.register_reader
    assert pyvista_module.has_scheme is _reg_mod.has_scheme
    assert pyvista_module.LocalFileRequiredError is _reg_mod.LocalFileRequiredError
    assert utilities_module.register_reader is _reg_mod.register_reader
    assert utilities_module.has_scheme is _reg_mod.has_scheme
    assert utilities_module.LocalFileRequiredError is _reg_mod.LocalFileRequiredError

    test_file = tmp_path / 'data.reloadext'
    test_file.touch()
    mock = MagicMock(return_value=pyvista_module.PolyData())
    pyvista_module.register_reader('.reloadext', mock)

    pyvista_result = pyvista_module.read(str(test_file))
    utilities_result = utilities_module.read(str(test_file))

    assert isinstance(pyvista_result, pyvista_module.PolyData)
    assert isinstance(utilities_result, pyvista_module.PolyData)
    assert mock.call_args_list == [
        call(str(test_file.resolve())),
        call(str(test_file.resolve())),
    ]
