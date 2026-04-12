"""Tests for custom Jupyter backend registration and discovery."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import pyvista as pv
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
    yield
    _custom_backends.clear()
    _custom_backends.update(original)


def _mock_handler(plotter, **kwargs):
    return {'plotter': plotter, **kwargs}


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
        'importlib.metadata.entry_points',
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
        pytest.warns(UserWarning, match='Available backends: "static", "wasm", "none"'),
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
