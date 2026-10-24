"""Tests for custom Jupyter backend registration and discovery."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import pytest

import pyvista as pv
from pyvista import jupyter as jupyter_mod
from pyvista.jupyter import ALLOWED_BACKENDS
from pyvista.jupyter import _custom_backend_sources
from pyvista.jupyter import _custom_backends
from pyvista.jupyter import _get_custom_backend_handler
from pyvista.jupyter import _resolve_backend
from pyvista.jupyter import _validate_jupyter_backend
from pyvista.jupyter import register_jupyter_backend
from pyvista.jupyter.notebook import _warn_missing_server_proxy
from pyvista.jupyter.notebook import handle_plotter

has_ipython = bool(importlib.util.find_spec('IPython'))
skip_no_ipython = pytest.mark.skipif(not has_ipython, reason='Requires IPython package')


@contextlib.contextmanager
def _without_custom_backends():
    """Run a test without any custom backend registrations or entry points."""
    saved = _custom_backends.copy()
    saved_sources = _custom_backend_sources.copy()
    saved_loaded = jupyter_mod._entry_points_loaded
    _custom_backends.clear()
    _custom_backend_sources.clear()
    jupyter_mod._entry_points_loaded = True
    try:
        yield
    finally:
        _custom_backends.clear()
        _custom_backends.update(saved)
        _custom_backend_sources.clear()
        _custom_backend_sources.update(saved_sources)
        jupyter_mod._entry_points_loaded = saved_loaded


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

    with (
        _without_custom_backends(),
        patch('pyvista.jupyter.entry_points', return_value=[mock_ep]),
    ):
        jupyter_mod._entry_points_loaded = False
        records = pv.registered_jupyter_backends()

    matches = [r for r in records if r.name == 'discovered_backend']
    assert len(matches) == 1
    assert matches[0].source == 'package.module:backend_func'


def test_entry_point_load_failure_warns_and_continues():
    broken = MagicMock()
    broken.name = 'broken_backend'
    broken.value = 'package:broken'
    broken.load.side_effect = RuntimeError('broken plugin')

    with (
        _without_custom_backends(),
        patch('pyvista.jupyter.entry_points', return_value=[broken]),
    ):
        jupyter_mod._entry_points_loaded = False
        with pytest.warns(
            UserWarning, match='Failed to load pyvista.jupyter_backends entry point'
        ):
            handler = _get_custom_backend_handler('broken_backend')
        assert handler is None


def test_entry_point_uses_renamed_group():
    """Confirm the entry-point group name is ``pyvista.jupyter_backends``."""
    captured: dict[str, object] = {}

    def fake_entry_points(*, group):
        captured['group'] = group
        return []

    with (
        _without_custom_backends(),
        patch('pyvista.jupyter.entry_points', side_effect=fake_entry_points),
    ):
        jupyter_mod._entry_points_loaded = False
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
    mock_ep = MagicMock()
    mock_ep.name = 'discovered'
    mock_ep.load.return_value = _mock_handler

    with (
        _without_custom_backends(),
        patch('pyvista.jupyter.entry_points', return_value=[mock_ep]),
    ):
        jupyter_mod._entry_points_loaded = False
        handler = _get_custom_backend_handler('discovered')
        assert handler is _mock_handler


@skip_no_ipython
def test_handle_plotter_falls_back_to_custom_backend_when_trame_unregistered():
    """When the requested trame backend has no handler but a custom backend exists, use it."""
    mock_handler = MagicMock(return_value='ep_widget')
    plotter = MagicMock()

    with _without_custom_backends():
        register_jupyter_backend('ep_backend', mock_handler)
        with pytest.warns(UserWarning, match='Using registered backend "ep_backend"'):
            result = handle_plotter(plotter, backend='trame')

    assert result == 'ep_widget'
    mock_handler.assert_called_once_with(plotter, screenshot=None)


@skip_no_ipython
def test_handle_plotter_static_fallback_lists_available_backends():
    """When trame has no handler and no custom backends exist, list available backends."""
    plotter = MagicMock()
    plotter.last_image = None

    with (
        _without_custom_backends(),
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
    """When trame-pyvista is installed, the entry-point trame backend is selected."""
    assert _resolve_backend() == 'trame'


@skip_no_ipython
def test_resolve_backend_prefers_custom_over_trame():
    """When both trame and a custom backend are available, prefer the user-registered one."""
    register_jupyter_backend('mybackend', _mock_handler)
    assert _resolve_backend() == 'mybackend'


@skip_no_ipython
def test_resolve_backend_prefers_custom_over_static():
    """When no entry points exist but a custom backend is registered, prefer it."""
    with _without_custom_backends():
        register_jupyter_backend('mybackend', _mock_handler)
        assert _resolve_backend() == 'mybackend'


@skip_no_ipython
def test_resolve_backend_falls_back_to_static():
    """When nothing else is available, _resolve_backend returns 'static'."""
    with _without_custom_backends():
        assert _resolve_backend() == 'static'


@skip_no_ipython
def test_handle_plotter_auto_selects_custom_backend():
    """When backend=None and only a custom backend is registered, auto-select it."""
    mock_handler = MagicMock(return_value='auto_widget')
    plotter = MagicMock()

    with _without_custom_backends():
        register_jupyter_backend('auto_backend', mock_handler)
        result = handle_plotter(plotter, backend=None)

    assert result == 'auto_widget'
    mock_handler.assert_called_once_with(plotter, screenshot=None)


@skip_no_ipython
def test_handle_plotter_auto_static_warns_install():
    """When backend=None and only static is available, warn about installing trame."""
    plotter = MagicMock()

    with (
        _without_custom_backends(),
        patch(
            'pyvista.jupyter.notebook.show_static_image',
            return_value='static_img',
        ),
        pytest.warns(UserWarning, match=r'pip install pyvista\[jupyter\]'),
    ):
        result = handle_plotter(plotter, backend=None)

    assert result == 'static_img'


@pytest.fixture
def set_server_proxy_importable(monkeypatch):
    """Control whether ``jupyter_server_proxy`` is importable and reset the warn-once cache."""
    find_spec = importlib.util.find_spec
    importable = False

    def fake_find_spec(name, *args, **kwargs):
        """Report ``jupyter_server_proxy`` as importable or not."""
        if name == 'jupyter_server_proxy':
            return MagicMock() if importable else None
        return find_spec(name, *args, **kwargs)

    def set_importable(value):
        """Set whether ``jupyter_server_proxy`` is importable."""
        nonlocal importable
        importable = value

    monkeypatch.setattr(importlib.util, 'find_spec', fake_find_spec)
    _warn_missing_server_proxy.cache_clear()
    yield set_importable
    _warn_missing_server_proxy.cache_clear()


@skip_no_ipython
@pytest.mark.parametrize(
    ('backend', 'theme_enabled', 'kwargs', 'installed', 'warns'),
    [
        ('trame', True, {}, False, True),
        ('server', True, {}, False, True),
        ('client', True, {}, False, True),
        ('html', True, {'mode': 'trame'}, False, True),
        ('trame', False, {'server_proxy_enabled': True}, False, True),
        ('trame', True, {'server_proxy_enabled': None}, False, True),
        ('trame', True, {'server_proxy_enabled': False}, False, False),
        ('trame', True, {'jupyter_extension_enabled': True}, False, False),
        ('trame', True, {'server_proxy_prefix': 'https://example.com/proxy/'}, False, False),
        ('trame', True, {'mode': 'html'}, False, False),
        ('trame', False, {}, False, False),
        ('trame', True, {}, True, False),
        ('html', True, {}, False, False),
    ],
)
def test_handle_plotter_warns_missing_server_proxy(
    monkeypatch, set_server_proxy_importable, backend, theme_enabled, kwargs, installed, warns
):
    """Warn when the trame iframe is routed through an unavailable jupyter-server-proxy."""
    set_server_proxy_importable(installed)
    monkeypatch.setattr(pv.global_theme.trame, '_server_proxy_enabled', theme_enabled)
    mock_handler = MagicMock(return_value='widget')
    plotter = MagicMock()

    with _without_custom_backends():
        register_jupyter_backend(backend, mock_handler, override=True)
        if warns:
            with pytest.warns(UserWarning, match='pip install jupyter-server-proxy'):
                result = handle_plotter(plotter, backend=backend, **kwargs)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                result = handle_plotter(plotter, backend=backend, **kwargs)

    assert result == 'widget'
    mock_handler.assert_called_once_with(plotter, screenshot=None, **kwargs)


@skip_no_ipython
@pytest.mark.usefixtures('set_server_proxy_importable')
def test_handle_plotter_warns_missing_server_proxy_once(monkeypatch):
    """Warn about a missing jupyter-server-proxy only on the first plot."""
    monkeypatch.setattr(pv.global_theme.trame, '_server_proxy_enabled', True)
    plotter = MagicMock()

    with _without_custom_backends():
        register_jupyter_backend('trame', MagicMock(), override=True)
        with pytest.warns(UserWarning, match='pip install jupyter-server-proxy'):
            handle_plotter(plotter, backend='trame')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            handle_plotter(plotter, backend='trame')


@skip_no_ipython
@pytest.mark.usefixtures('set_server_proxy_importable')
def test_handle_plotter_static_fallback_skips_server_proxy_warning(monkeypatch):
    """Do not warn about jupyter-server-proxy when the plot falls back to a static image."""
    monkeypatch.setattr(pv.global_theme.trame, '_server_proxy_enabled', True)

    with (
        _without_custom_backends(),
        patch('pyvista.jupyter.notebook.show_static_image', return_value='static_img'),
        pytest.warns(UserWarning, match='Falling back to a static output') as record,
    ):
        result = handle_plotter(MagicMock(), backend='trame')

    assert result == 'static_img'
    assert not any('jupyter-server-proxy' in str(w.message) for w in record)


def test_allowed_backends():
    """``ALLOWED_BACKENDS`` holds the backend names in order."""
    assert ALLOWED_BACKENDS == ('static', 'client', 'server', 'trame', 'html', 'none')


@pytest.mark.parametrize(
    ('ipykernel', 'marimo', 'expected'),
    [
        (True, None, True),
        (False, None, False),
        (False, True, True),
        (False, False, False),
        (False, 'missing', False),
    ],
)
def test_in_notebook(monkeypatch, ipykernel, marimo, expected):
    monkeypatch.setattr(jupyter_mod.scooby, 'in_ipykernel', lambda: ipykernel)
    if marimo is None:
        monkeypatch.delitem(sys.modules, 'marimo', raising=False)
    elif marimo == 'missing':
        monkeypatch.setitem(sys.modules, 'marimo', SimpleNamespace())
    else:
        monkeypatch.setitem(
            sys.modules, 'marimo', SimpleNamespace(running_in_notebook=lambda: marimo)
        )
    assert jupyter_mod._in_notebook() is expected


def _fake_marimo(monkeypatch):
    """Run as a marimo notebook outside a Jupyter kernel and return the fake marimo module."""
    marimo = SimpleNamespace(running_in_notebook=lambda: True, output=MagicMock())
    monkeypatch.setattr(jupyter_mod.scooby, 'in_ipykernel', lambda: False)
    monkeypatch.setitem(sys.modules, 'marimo', marimo)
    return marimo


@pytest.mark.parametrize(('theme', 'expected'), [(None, True), (False, False)])
def test_plotter_notebook_in_marimo(monkeypatch, theme, expected):
    _fake_marimo(monkeypatch)
    monkeypatch.setattr(pv.global_theme, 'notebook', theme)
    pl = pv.Plotter()
    assert pl.notebook is expected
    pl.close()


def test_display_through_marimo_without_ipython(monkeypatch):
    marimo = _fake_marimo(monkeypatch)
    monkeypatch.setitem(sys.modules, 'IPython', None)
    jupyter_mod._display('image')
    marimo.output.append.assert_called_once_with('image')


def test_display_without_ipython_raises(monkeypatch):
    monkeypatch.delitem(sys.modules, 'marimo', raising=False)
    monkeypatch.setitem(sys.modules, 'IPython', None)
    with pytest.raises(ImportError, match='Install IPython'):
        jupyter_mod._display('image')


@pytest.mark.parametrize(('renders', 'expected'), [(True, 'trame'), (False, 'static')])
def test_resolve_backend_in_marimo(monkeypatch, renders, expected):
    _fake_marimo(monkeypatch)
    monkeypatch.setattr(jupyter_mod, '_marimo_renders_trame_pyvista', lambda: renders)
    with _without_custom_backends():
        for name in ('trame', 'html'):
            _custom_backends[name] = _mock_handler
            _custom_backend_sources[name] = 'trame_pyvista.jupyter:show_trame'
        assert _resolve_backend() == expected


def test_handle_plotter_in_marimo_asks_to_upgrade_trame_pyvista(monkeypatch):
    _fake_marimo(monkeypatch)
    monkeypatch.setattr(jupyter_mod, '_marimo_renders_trame_pyvista', lambda: False)
    with (
        _without_custom_backends(),
        patch('pyvista.jupyter.notebook.show_static_image', return_value='static_img'),
    ):
        _custom_backends['trame'] = _mock_handler
        _custom_backend_sources['trame'] = 'trame_pyvista.jupyter:show_trame'
        with pytest.warns(UserWarning, match='Upgrade trame-pyvista'):
            result = handle_plotter(MagicMock())
    assert result == 'static_img'


def test_resolve_backend_in_marimo_keeps_other_plugins(monkeypatch):
    _fake_marimo(monkeypatch)
    monkeypatch.setattr(jupyter_mod, '_marimo_renders_trame_pyvista', lambda: False)
    with _without_custom_backends():
        _custom_backends['trame'] = _mock_handler
        _custom_backend_sources['trame'] = 'trame_pyvista.jupyter:show_trame'
        _custom_backends['other'] = _mock_handler
        _custom_backend_sources['other'] = 'other_plugin:show'
        assert _resolve_backend() == 'other'
