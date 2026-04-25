"""Tests for custom Jupyter backend registration and discovery."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import pytest

import pyvista as pv
from pyvista import jupyter as jupyter_mod
from pyvista.jupyter import _custom_backend_sources
from pyvista.jupyter import _custom_backends
from pyvista.jupyter import _get_custom_backend_handler
from pyvista.jupyter import _resolve_backend
from pyvista.jupyter import _validate_jupyter_backend
from pyvista.jupyter import register_jupyter_backend
from pyvista.jupyter.notebook import handle_plotter

has_ipython = bool(importlib.util.find_spec('IPython'))
skip_no_ipython = pytest.mark.skipif(not has_ipython, reason='Requires IPython package')


@contextlib.contextmanager
def _block_trame_import():
    """Temporarily make ``from pyvista.trame.jupyter import …`` raise ImportError."""
    trame_keys = [k for k in sys.modules if k.startswith('pyvista.trame')]
    saved = {k: sys.modules.pop(k) for k in trame_keys}
    sentinel = dict.fromkeys(trame_keys) | {'pyvista.trame.jupyter': None}
    try:
        with patch.dict(sys.modules, sentinel):
            yield
    finally:
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def _clean_custom_backends():
    """Remove any custom backends registered during tests."""
    original = _custom_backends.copy()
    original_sources = _custom_backend_sources.copy()
    original_loaded = jupyter_mod._entry_points_loaded
    yield
    _custom_backends.clear()
    _custom_backends.update(original)
    _custom_backend_sources.clear()
    _custom_backend_sources.update(original_sources)
    jupyter_mod._entry_points_loaded = original_loaded


def _mock_handler(plotter, **kwargs):
    return {'plotter': plotter, **kwargs}


def _replacement_handler(_plotter, **_kwargs):
    return {'replaced': True}


@skip_no_ipython
def test_register_and_validate():
    register_jupyter_backend('mybackend', _mock_handler)
    # Should not raise
    result = _validate_jupyter_backend('mybackend')
    assert result == 'mybackend'


@skip_no_ipython
def test_register_case_insensitive():
    register_jupyter_backend('MyBackend', _mock_handler)
    assert _get_custom_backend_handler('mybackend') is _mock_handler


@pytest.mark.parametrize('name', ['static', 'trame', 'server', 'client', 'html', 'none'])
def test_register_builtin_collision(name):
    with pytest.raises(ValueError, match='collides with built-in backend'):
        register_jupyter_backend(name, _mock_handler)


@skip_no_ipython
def test_register_builtin_override_allowed():
    register_jupyter_backend('static', _mock_handler, override=True)
    assert _get_custom_backend_handler('static') is _mock_handler


@skip_no_ipython
def test_register_custom_collision_warns_and_replaces():
    register_jupyter_backend('mycollide', _mock_handler)
    with pytest.warns(UserWarning, match='replaces an existing custom registration'):
        register_jupyter_backend('mycollide', _replacement_handler)
    assert _get_custom_backend_handler('mycollide') is _replacement_handler


@skip_no_ipython
def test_register_custom_collision_override_silent():
    register_jupyter_backend('mycollide', _mock_handler)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        register_jupyter_backend('mycollide', _replacement_handler, override=True)
    assert _get_custom_backend_handler('mycollide') is _replacement_handler


def test_registered_jupyter_backends_returns_record_with_source():
    register_jupyter_backend('demo_backend', _mock_handler)
    records = pv.registered_jupyter_backends()
    matches = [r for r in records if r.name == 'demo_backend']
    assert len(matches) == 1
    record = matches[0]
    assert record.handler is _mock_handler
    assert record.source.endswith('_mock_handler')


def test_registered_jupyter_backends_includes_entry_point_source():
    jupyter_mod._entry_points_loaded = False

    mock_ep = MagicMock()
    mock_ep.name = 'discovered_backend'
    mock_ep.value = 'package.module:backend_func'
    mock_ep.load.return_value = _mock_handler

    with patch('pyvista.jupyter.entry_points', return_value=[mock_ep]):
        records = pv.registered_jupyter_backends()

    matches = [r for r in records if r.name == 'discovered_backend']
    assert len(matches) == 1
    assert matches[0].source == 'package.module:backend_func'


def test_entry_point_load_failure_warns_and_continues():
    jupyter_mod._entry_points_loaded = False

    broken = MagicMock()
    broken.name = 'broken_backend'
    broken.value = 'package:broken'
    broken.load.side_effect = RuntimeError('broken plugin')

    with (
        patch('pyvista.jupyter.entry_points', return_value=[broken]),
        pytest.warns(UserWarning, match='Failed to load pyvista.jupyter_backends entry point'),
    ):
        assert _get_custom_backend_handler('broken_backend') is None


def test_entry_point_uses_renamed_group():
    """Confirm the entry-point group name is ``pyvista.jupyter_backends``."""
    jupyter_mod._entry_points_loaded = False

    captured: dict[str, object] = {}

    def fake_entry_points(*, group):
        captured['group'] = group
        return []

    with patch('pyvista.jupyter.entry_points', side_effect=fake_entry_points):
        jupyter_mod._ensure_entry_points()

    assert captured['group'] == 'pyvista.jupyter_backends'


@skip_no_ipython
def test_get_custom_backend_handler_returns_none_for_unknown():
    assert _get_custom_backend_handler('nonexistent') is None


@skip_no_ipython
def test_validate_lists_custom_backends_in_error():
    register_jupyter_backend('custom1', _mock_handler)
    with pytest.raises(ValueError, match='custom1'):
        _validate_jupyter_backend('totally_invalid')


@skip_no_ipython
def test_set_jupyter_backend_custom():
    register_jupyter_backend('testbackend', _mock_handler)
    pv.set_jupyter_backend('testbackend')
    assert pv.global_theme.jupyter_backend == 'testbackend'
    # Reset
    pv.set_jupyter_backend(None)


@skip_no_ipython
def test_handle_plotter_dispatches_custom():
    mock_handler = MagicMock(return_value='widget')
    register_jupyter_backend('mock', mock_handler)

    plotter = MagicMock()
    result = handle_plotter(plotter, backend='mock', foo='bar')

    mock_handler.assert_called_once_with(plotter, screenshot=None, foo='bar')
    assert result == 'widget'


@skip_no_ipython
def test_entry_point_discovery():
    import pyvista.jupyter as jupyter_mod

    # Reset discovery state
    jupyter_mod._entry_points_loaded = False

    mock_ep = MagicMock()
    mock_ep.name = 'discovered'
    mock_ep.load.return_value = _mock_handler

    with patch(
        'pyvista.jupyter.entry_points',
        return_value=[mock_ep],
    ):
        handler = _get_custom_backend_handler('discovered')
        assert handler is _mock_handler


@skip_no_ipython
def test_handle_plotter_falls_back_to_custom_backend_on_trame_import_error():
    """When trame is unavailable but an entry-point backend is registered, use it."""
    mock_handler = MagicMock(return_value='ep_widget')
    register_jupyter_backend('ep_backend', mock_handler)

    plotter = MagicMock()

    with (
        _block_trame_import(),
        pytest.warns(UserWarning, match='Using registered backend "ep_backend"'),
    ):
        result = handle_plotter(plotter, backend='trame')

    assert result == 'ep_widget'
    mock_handler.assert_called_once_with(plotter, screenshot=None)


@skip_no_ipython
def test_handle_plotter_static_fallback_lists_available_backends():
    """When trame is unavailable and no custom backends exist, list available backends."""
    plotter = MagicMock()
    plotter.last_image = None

    with (
        _block_trame_import(),
        patch(
            'pyvista.jupyter.notebook.show_static_image',
            return_value='static_img',
        ) as mock_static,
        pytest.warns(UserWarning, match='Available backends: "static", "none"'),
    ):
        result = handle_plotter(plotter, backend='trame')

    assert result == 'static_img'
    mock_static.assert_called_once()


@skip_no_ipython
def test_resolve_backend_prefers_trame_when_no_custom():
    """When trame is available and no custom backends registered, returns 'trame'."""
    assert _resolve_backend() == 'trame'


@skip_no_ipython
def test_resolve_backend_prefers_custom_over_trame():
    """When both trame and a custom backend are available, prefer the custom one."""
    register_jupyter_backend('mybackend', _mock_handler)
    assert _resolve_backend() == 'mybackend'


@skip_no_ipython
def test_resolve_backend_prefers_custom_over_static():
    """When trame is unavailable but a custom backend is registered, prefer it."""
    register_jupyter_backend('mybackend', _mock_handler)
    with _block_trame_import():
        assert _resolve_backend() == 'mybackend'


@skip_no_ipython
def test_resolve_backend_falls_back_to_static():
    """When nothing else is available, _resolve_backend returns 'static'."""
    with _block_trame_import():
        assert _resolve_backend() == 'static'


@skip_no_ipython
def test_handle_plotter_auto_selects_custom_backend():
    """When backend=None, trame unavailable, registered backend is auto-selected."""
    mock_handler = MagicMock(return_value='auto_widget')
    register_jupyter_backend('auto_backend', mock_handler)

    plotter = MagicMock()

    with _block_trame_import():
        result = handle_plotter(plotter, backend=None)

    assert result == 'auto_widget'
    mock_handler.assert_called_once_with(plotter, screenshot=None)


@skip_no_ipython
def test_handle_plotter_auto_static_warns_install():
    """When backend=None and only static is available, warn about installing trame."""
    plotter = MagicMock()

    with (
        _block_trame_import(),
        patch(
            'pyvista.jupyter.notebook.show_static_image',
            return_value='static_img',
        ),
        pytest.warns(UserWarning, match=r'pip install "pyvista\[jupyter\]"'),
    ):
        result = handle_plotter(plotter, backend=None)

    assert result == 'static_img'
