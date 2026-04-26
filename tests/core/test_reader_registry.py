"""Tests for custom reader registration and discovery."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import re
import subprocess
import sys
import types
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import pytest

import pyvista as pv
from pyvista.core.utilities import reader_registry as _reg_mod


@pytest.fixture(autouse=True)
def _clean_custom_readers():
    """Restore the reader registry and clean temp files around every test."""
    state = _reg_mod._save_registry_state()
    orig_temp_files = list(_reg_mod._temp_files)
    yield
    _reg_mod._temp_files[:] = [
        path for path in _reg_mod._temp_files if path not in orig_temp_files
    ]
    _reg_mod._cleanup_temp_files()
    _reg_mod._temp_files.extend(orig_temp_files)
    _reg_mod._restore_registry_state(state)


def _mock_reader(_path, **__):
    return pv.PolyData()


def _load_module_copy(module_name: str, module_path: str | Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_import_registers_cleanup_handler():
    with patch('atexit.register') as register:
        module = _load_module_copy('reader_registry_test_copy', _reg_mod.__file__)

    register.assert_called_once_with(module._cleanup_temp_files)
    assert module._custom_ext_readers == {}
    assert module._custom_ext_reader_sources == {}
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


def test_entry_point_load_failure_warns_and_returns_none():
    _reg_mod._entry_points_loaded = False

    broken = MagicMock()
    broken.name = '.broken'
    broken.value = 'package:broken'
    broken.load.side_effect = RuntimeError('broken plugin')

    with (
        patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[broken]),
        patch('pyvista.core.utilities.reader_registry.warn_external') as warn,
    ):
        assert _reg_mod._get_ext_handler('.broken') is None

    warn.assert_called_once()
    message = warn.call_args.args[0]
    assert 'Failed to load pyvista.readers entry point' in message
    assert 'package:broken' in message
    assert 'broken plugin' in message


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


def test_top_level_exports_available_in_fresh_python_process():
    command = (
        'import pyvista as pv; '
        'import pyvista.core.utilities as utilities; '
        'assert pv.register_reader is utilities.register_reader; '
        'assert pv.has_scheme is utilities.has_scheme; '
        'assert pv.LocalFileRequiredError is utilities.LocalFileRequiredError'
    )
    subprocess.run(
        [sys.executable, '-c', command],
        check=True,
        capture_output=True,
        text=True,
    )


def test_registered_readers_returns_record_with_source():
    pv.register_reader('.mything', _mock_reader)
    records = pv.registered_readers()
    matches = [r for r in records if r.extension == '.mything']
    assert len(matches) == 1
    record = matches[0]
    assert record.handler is _mock_reader
    assert record.source.endswith('_mock_reader')


def test_registered_readers_includes_entry_point_source():
    _reg_mod._entry_points_loaded = False

    mock_ep = MagicMock()
    mock_ep.name = '.discovered_src'
    mock_ep.value = 'package.module:reader_func'
    mock_ep.load.return_value = _mock_reader

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        records = pv.registered_readers()

    matches = [r for r in records if r.extension == '.discovered_src']
    assert len(matches) == 1
    assert matches[0].source == 'package.module:reader_func'


def test_register_custom_collision_warns_and_replaces():
    pv.register_reader('.collide', _mock_reader)

    def replacement(_path, **__):
        return pv.PolyData()

    with pytest.warns(UserWarning, match='replaces an existing custom reader'):
        pv.register_reader('.collide', replacement)
    assert _reg_mod._custom_ext_readers['.collide'] is replacement


def test_register_custom_collision_override_silent():
    pv.register_reader('.collide', _mock_reader)

    def replacement(_path, **__):
        return pv.PolyData()

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pv.register_reader('.collide', replacement, override=True)
    assert _reg_mod._custom_ext_readers['.collide'] is replacement


def test_metadata_scan_does_not_load_plugin():
    """``_ensure_entry_points`` records metadata without importing plugins."""
    _reg_mod._entry_points_loaded = False
    _reg_mod._pending_ext_readers.clear()

    mock_ep = MagicMock()
    mock_ep.name = '.lazy'
    mock_ep.value = 'lazy_pkg:reader'

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        _reg_mod._ensure_entry_points()

    mock_ep.load.assert_not_called()
    assert '.lazy' in _reg_mod._pending_ext_readers
    assert '.lazy' not in _reg_mod._custom_ext_readers


def test_unrelated_extension_does_not_load_plugin():
    """Looking up an extension *no plugin claims* never imports any plugin."""
    _reg_mod._entry_points_loaded = False
    _reg_mod._pending_ext_readers.clear()

    plugin_ep = MagicMock()
    plugin_ep.name = '.someplugin'
    plugin_ep.value = 'someplugin:reader'

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[plugin_ep]):
        # Built-in extensions like ``.vtp`` resolve via the VTK class
        # readers, never via the entry-point registry. The lookup must
        # not load any plugin.
        assert _reg_mod._get_ext_handler('.vtp') is None

    plugin_ep.load.assert_not_called()
    # Pending plugin extension is still recorded for later resolution
    assert '.someplugin' in _reg_mod._pending_ext_readers


def test_matching_extension_loads_only_its_plugin():
    """Looking up an extension a plugin claims loads *only* that plugin."""
    _reg_mod._entry_points_loaded = False
    _reg_mod._pending_ext_readers.clear()

    wanted = MagicMock()
    wanted.name = '.wanted'
    wanted.value = 'wanted_pkg:reader'
    wanted.load.return_value = _mock_reader

    other = MagicMock()
    other.name = '.other'
    other.value = 'other_pkg:reader'

    with patch(
        'pyvista.core.utilities.reader_registry.entry_points',
        return_value=[wanted, other],
    ):
        handler = _reg_mod._get_ext_handler('.wanted')

    assert handler is _mock_reader
    wanted.load.assert_called_once()
    other.load.assert_not_called()
    # The other plugin remains pending — still discoverable, still not loaded
    assert '.other' in _reg_mod._pending_ext_readers


def test_list_custom_exts_includes_pending_without_loading():
    """``_list_custom_exts`` reports pending extensions without loading."""
    _reg_mod._entry_points_loaded = False
    _reg_mod._pending_ext_readers.clear()

    mock_ep = MagicMock()
    mock_ep.name = '.discoverable'
    mock_ep.value = 'discoverable_pkg:reader'

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        exts = _reg_mod._list_custom_exts()

    assert '.discoverable' in exts
    mock_ep.load.assert_not_called()


def test_registered_readers_forces_full_discovery():
    """``registered_readers`` resolves every pending plugin so callers see all."""
    _reg_mod._entry_points_loaded = False
    _reg_mod._pending_ext_readers.clear()

    mock_ep = MagicMock()
    mock_ep.name = '.eager'
    mock_ep.value = 'eager_pkg:reader'
    mock_ep.load.return_value = _mock_reader

    with patch('pyvista.core.utilities.reader_registry.entry_points', return_value=[mock_ep]):
        records = pv.registered_readers()

    matches = [r for r in records if r.extension == '.eager']
    assert len(matches) == 1
    assert matches[0].handler is _mock_reader
    mock_ep.load.assert_called_once()
