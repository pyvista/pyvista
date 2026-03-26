"""Tests for custom Jupyter backend registration and discovery."""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import pyvista as pv
from pyvista.jupyter import _custom_backends
from pyvista.jupyter import _get_custom_backend_handler
from pyvista.jupyter import _validate_jupyter_backend
from pyvista.jupyter import register_jupyter_backend

has_ipython = bool(importlib.util.find_spec('IPython'))
skip_no_ipython = pytest.mark.skipif(not has_ipython, reason='Requires IPython package')


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
    from pyvista.jupyter.notebook import handle_plotter

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
