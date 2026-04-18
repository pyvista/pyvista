"""Tests for custom writer registration and discovery."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import pyvista as pv
from pyvista.core.utilities import writer_registry as _reg_mod


@pytest.fixture(autouse=True)
def _clean_custom_writers():
    """Restore the writer registry around every test."""
    state = _reg_mod._save_registry_state()
    yield
    _reg_mod._restore_registry_state(state)


def _touching_writer(_dataset, path):
    """Minimal writer stand-in that creates an empty file at *path*."""
    Path(path).touch()


def _noop_writer(_dataset, _path):
    """Writer stand-in that returns without touching disk."""


def test_initial_module_state():
    assert _reg_mod._custom_ext_writers == {}


def test_register_extension():
    pv.register_writer('.myformat', _noop_writer)
    assert _reg_mod._custom_ext_writers['.myformat'] is _noop_writer


def test_register_extension_without_dot():
    pv.register_writer('myformat', _noop_writer)
    assert '.myformat' in _reg_mod._custom_ext_writers


def test_register_extension_case_insensitive():
    pv.register_writer('.MyFormat', _noop_writer)
    assert '.myformat' in _reg_mod._custom_ext_writers


def test_register_builtin_collision():
    with pytest.raises(ValueError, match='collides with a built-in PyVista writer'):
        pv.register_writer('.vtp', _noop_writer)


def test_register_builtin_override():
    pv.register_writer('.vtp', _noop_writer, override=True)
    assert _reg_mod._custom_ext_writers['.vtp'] is _noop_writer


def test_register_as_decorator():
    @pv.register_writer('.decorated')
    def my_writer(_dataset, _path):
        return None

    assert _reg_mod._custom_ext_writers['.decorated'] is my_writer


def test_register_decorator_stacked():
    @pv.register_writer('.ext1')
    @pv.register_writer('.ext2')
    def my_writer(_dataset, _path):
        return None

    assert _reg_mod._custom_ext_writers['.ext1'] is my_writer
    assert _reg_mod._custom_ext_writers['.ext2'] is my_writer


def test_get_ext_handler():
    pv.register_writer('.myformat', _noop_writer)
    assert _reg_mod._get_ext_handler('.myformat') is _noop_writer


def test_get_ext_handler_returns_none():
    assert _reg_mod._get_ext_handler('.nonexistent') is None


def test_builtin_writer_exts_includes_common_formats():
    exts = _reg_mod._get_builtin_writer_exts()
    assert {'.vtp', '.vtu', '.vti', '.vtm'} <= exts


def test_entry_point_discovery_extension():
    mock_ep = MagicMock()
    mock_ep.name = '.discovered'
    mock_ep.value = 'package:ep'
    mock_ep.load.return_value = _noop_writer

    with patch('pyvista.core.utilities.writer_registry.entry_points', return_value=[mock_ep]):
        assert _reg_mod._get_ext_handler('.discovered') is _noop_writer


def test_entry_point_name_without_dot_is_normalized():
    mock_ep = MagicMock()
    mock_ep.name = 'discovered'
    mock_ep.value = 'package:ep'
    mock_ep.load.return_value = _noop_writer

    with patch('pyvista.core.utilities.writer_registry.entry_points', return_value=[mock_ep]):
        assert _reg_mod._get_ext_handler('.discovered') is _noop_writer


def test_entry_point_duplicate_warns_and_uses_first_handler():
    first = MagicMock()
    first.name = 'discovered'
    first.value = 'package:first'
    first.load.return_value = _noop_writer

    second = MagicMock()
    second.name = 'discovered'
    second.value = 'package:second'

    with (
        patch(
            'pyvista.core.utilities.writer_registry.entry_points',
            return_value=[first, second],
        ),
        patch('pyvista.core.utilities.writer_registry.warn_external') as warn,
    ):
        handler = _reg_mod._get_ext_handler('.discovered')

    assert handler is _noop_writer
    second.load.assert_not_called()
    warn.assert_called_once()
    assert warn.call_args.args[0] == (
        'Multiple pyvista.writers entry points registered for ".discovered": '
        'package:first, package:second. Using package:first.'
    )


def test_entry_point_does_not_override_explicit():
    pv.register_writer('.myext', _noop_writer)

    other_handler = MagicMock()
    mock_ep = MagicMock()
    mock_ep.name = '.myext'
    mock_ep.value = 'package:ep'
    mock_ep.load.return_value = other_handler

    with patch('pyvista.core.utilities.writer_registry.entry_points', return_value=[mock_ep]):
        assert _reg_mod._get_ext_handler('.myext') is _noop_writer
    other_handler.assert_not_called()


def test_entry_point_load_failure_warns_and_returns_none():
    broken = MagicMock()
    broken.name = '.broken'
    broken.value = 'package:broken'
    broken.load.side_effect = RuntimeError('broken plugin')

    with (
        patch('pyvista.core.utilities.writer_registry.entry_points', return_value=[broken]),
        patch('pyvista.core.utilities.writer_registry.warn_external') as warn,
    ):
        assert _reg_mod._get_ext_handler('.broken') is None

    warn.assert_called_once()
    message = warn.call_args.args[0]
    assert 'Failed to load pyvista.writers entry point' in message
    assert 'package:broken' in message
    assert 'broken plugin' in message


def test_entry_points_loaded_persists_across_lookups():
    with patch(
        'pyvista.core.utilities.writer_registry.entry_points', return_value=[]
    ) as entry_points_mock:
        _reg_mod._get_ext_handler('.nothing')
        _reg_mod._get_ext_handler('.nothing-else')
    entry_points_mock.assert_called_once_with(group='pyvista.writers')


def test_save_registry_state_snapshots_entry_points_loaded():
    state = _reg_mod._save_registry_state()
    assert state['entry_points_loaded'] is _reg_mod._entry_points_loaded

    # Mutate the module then restore; the flag must round-trip.
    original = _reg_mod._entry_points_loaded
    _reg_mod._entry_points_loaded = not original
    _reg_mod._restore_registry_state(state)
    assert _reg_mod._entry_points_loaded is original


def test_save_with_custom_extension(tmp_path):
    target = tmp_path / 'mesh.myext'
    mock = MagicMock(side_effect=_touching_writer)
    pv.register_writer('.myext', mock)

    sphere = pv.Sphere()
    sphere.save(target)

    mock.assert_called_once()
    args, kwargs = mock.call_args
    assert args[0] is sphere
    assert args[1] == str(target.resolve())
    assert kwargs == {}
    assert target.exists()


def test_save_custom_writer_dataset_first_argument_order(tmp_path):
    target = tmp_path / 'mesh.myext'
    received: dict[str, object] = {}

    def writer(dataset, path):
        received['dataset'] = dataset
        received['path'] = path
        Path(path).touch()

    pv.register_writer('.myext', writer)
    sphere = pv.Sphere()
    sphere.save(target)

    assert received['dataset'] is sphere
    assert received['path'] == str(target.resolve())


def test_save_override_replaces_builtin(tmp_path):
    # override=True both registers AND has the custom writer replace the
    # built-in writer at dispatch time, mirroring how the reader registry
    # dispatches custom readers over built-ins.
    calls: list[str] = []

    def custom(_ds, path):
        calls.append(path)
        Path(path).touch()

    pv.register_writer('.vtp', custom, override=True)

    target = tmp_path / 'mesh.vtp'
    pv.Sphere().save(target)

    assert calls == [str(target.resolve())]
    # File created by our custom writer (empty), not the built-in XML
    # writer which would have produced a non-trivial file.
    assert target.stat().st_size == 0


def test_save_builtin_used_when_no_custom_registered(tmp_path):
    target = tmp_path / 'mesh.vtp'
    pv.Sphere().save(target)
    # Built-in writer produces a non-empty XML file.
    assert target.stat().st_size > 0


def test_save_custom_dispatch_for_non_builtin_extension(tmp_path):
    target = tmp_path / 'mesh.customext'
    mock = MagicMock(side_effect=_touching_writer)
    pv.register_writer('.customext', mock)

    pv.Sphere().save(target)
    mock.assert_called_once()


def test_save_invalid_extension_error_lists_custom_exts(tmp_path):
    pv.register_writer('.registered', _noop_writer)

    target = tmp_path / 'mesh.bogus'
    with pytest.raises(ValueError, match=r'\.registered'):
        pv.Sphere().save(target)


def test_save_custom_writer_missing_parent_dir(tmp_path):
    mock = MagicMock(side_effect=_touching_writer)
    pv.register_writer('.myext', mock)

    target = tmp_path / 'missing_dir' / 'mesh.myext'
    with pytest.raises(FileNotFoundError, match='Parent directory'):
        pv.Sphere().save(target)
    mock.assert_not_called()


def test_save_custom_writer_failed_to_create_file(tmp_path):
    # Handler does nothing — existence check should raise OSError.
    pv.register_writer('.silent', _noop_writer)

    target = tmp_path / 'mesh.silent'
    with pytest.raises(OSError, match='Custom writer failed to write file'):
        pv.Sphere().save(target)


def test_save_forwards_writer_kwargs_to_custom_handler(tmp_path):
    target = tmp_path / 'mesh.myext'
    received: dict[str, object] = {}

    def writer(_dataset, path, *, level, threads=1):
        received['level'] = level
        received['threads'] = threads
        Path(path).touch()

    pv.register_writer('.myext', writer)
    pv.Sphere().save(target, level=9, threads=4)

    assert received == {'level': 9, 'threads': 4}


def test_save_custom_handler_without_kwargs_still_works(tmp_path):
    target = tmp_path / 'mesh.myext'

    # Handler does not declare **kwargs and we pass no extras — must not raise.
    def writer(_dataset, path):
        Path(path).touch()

    pv.register_writer('.myext', writer)
    pv.Sphere().save(target)
    assert target.exists()


def test_save_custom_handler_without_kwargs_rejects_extras(tmp_path):
    target = tmp_path / 'mesh.myext'

    def writer(_dataset, path):
        Path(path).touch()  # pragma: no cover — raises before reaching here

    pv.register_writer('.myext', writer)
    # Python itself raises TypeError when the handler does not accept kwargs.
    with pytest.raises(TypeError, match="unexpected keyword argument 'level'"):
        pv.Sphere().save(target, level=9)


def test_save_builtin_writer_rejects_unexpected_kwargs(tmp_path):
    target = tmp_path / 'mesh.vtp'
    with pytest.raises(
        TypeError,
        match=(
            r"unexpected keyword arguments \['level'\] for "
            r"built-in VTK writer for extension '\.vtp'"
        ),
    ):
        pv.Sphere().save(target, level=9)
    # Strict rejection — file must not have been written.
    assert not target.exists()


def test_save_builtin_writer_rejects_multiple_unexpected_kwargs(tmp_path):
    target = tmp_path / 'mesh.vtp'
    with pytest.raises(
        TypeError,
        match=r"unexpected keyword arguments \['level', 'threads'\]",
    ):
        pv.Sphere().save(target, level=9, threads=4)


def test_save_pickle_rejects_unexpected_kwargs(tmp_path):
    target = tmp_path / 'mesh.pkl'
    with pytest.raises(
        TypeError,
        match=(
            r"unexpected keyword arguments \['level'\] for "
            r"pickle format for extension '\.pkl'"
        ),
    ):
        pv.Sphere().save(target, level=9)
    assert not target.exists()


def test_save_override_custom_writer_receives_kwargs(tmp_path):
    # override=True path still forwards kwargs to the custom writer.
    received: dict[str, object] = {}

    def custom(_ds, path, *, level):
        received['level'] = level
        Path(path).touch()

    pv.register_writer('.vtp', custom, override=True)
    pv.Sphere().save(tmp_path / 'mesh.vtp', level=22)
    assert received == {'level': 22}


def test_top_level_exports_available_in_fresh_python_process():
    command = (
        'import pyvista as pv; '
        'import pyvista.core.utilities as utilities; '
        'assert pv.register_writer is utilities.register_writer'
    )
    subprocess.run(
        [sys.executable, '-c', command],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
