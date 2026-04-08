"""Tests for interactor style registration and discovery."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting import interactor_style_registry as _reg_mod
from pyvista.plotting.interactor_style_registry import _get_interactor_style_handler


def _mock_style_factory(_interactor):
    return _vtk.vtkInteractorStyleTrackballCamera()


def test_register_interactor_style_case_insensitive():
    pv.register_interactor_style('MyStyle', _mock_style_factory)

    assert _get_interactor_style_handler('mystyle') is _mock_style_factory


def test_register_interactor_style_rejects_empty_name():
    with pytest.raises(ValueError, match='must not be empty'):
        pv.register_interactor_style('', _mock_style_factory)


def test_register_interactor_style_rejects_builtin_collision():
    with pytest.raises(ValueError, match='collides with a built-in interactor style'):
        pv.register_interactor_style('terrain_style', _mock_style_factory)


def test_available_interactor_style_names_includes_builtins_and_custom():
    pv.register_interactor_style('my_custom', _mock_style_factory)

    names = _reg_mod._available_interactor_style_names()

    assert 'trackball_style' in names
    assert 'terrain_style' in names
    assert 'my_custom' in names
    assert names == tuple(sorted(names))


def test_entry_point_discovery_is_cached():
    mock_entry_point = MagicMock()
    mock_entry_point.name = 'discovered_style'
    mock_entry_point.value = 'my_package:custom_style'
    mock_entry_point.load.return_value = _mock_style_factory

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.interactor_style_registry.entry_points',
        return_value=[mock_entry_point],
    ):
        assert _reg_mod._has_interactor_style('discovered_style') is True
        mock_entry_point.load.assert_called_once()

        handler = _get_interactor_style_handler('discovered_style')
        assert handler is _mock_style_factory
        mock_entry_point.load.assert_called_once()

        cached_handler = _get_interactor_style_handler('discovered_style')
        assert cached_handler is _mock_style_factory
        mock_entry_point.load.assert_called_once()


def test_entry_point_does_not_override_explicit_registration():
    other_handler = MagicMock(return_value=_vtk.vtkInteractorStyleTrackballCamera())
    pv.register_interactor_style('explicit_style', _mock_style_factory)

    mock_entry_point = MagicMock()
    mock_entry_point.name = 'explicit_style'
    mock_entry_point.value = 'my_package:other_style'
    mock_entry_point.load.return_value = other_handler

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.interactor_style_registry.entry_points',
        return_value=[mock_entry_point],
    ):
        handler = _get_interactor_style_handler('explicit_style')

    assert handler is _mock_style_factory
    mock_entry_point.load.assert_not_called()


def test_entry_point_mapping_discovery():
    mock_entry_point = MagicMock()
    mock_entry_point.name = 'pyvista_interactors'
    mock_entry_point.value = 'my_package:INTERACTOR_STYLES'
    mock_entry_point.load.return_value = {
        'discovered_style': _mock_style_factory,
    }

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.interactor_style_registry.entry_points',
        return_value=[mock_entry_point],
    ):
        assert _reg_mod._has_interactor_style('discovered_style') is True

        handler = _get_interactor_style_handler('discovered_style')

    assert handler is _mock_style_factory


def test_entry_point_load_failure_warns():
    mock_entry_point = MagicMock()
    mock_entry_point.name = 'broken_style'
    mock_entry_point.value = 'my_package:broken'
    mock_entry_point.load.side_effect = ImportError('no such module')

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.interactor_style_registry.entry_points',
            return_value=[mock_entry_point],
        ),
        pytest.warns(UserWarning, match='Failed to load'),
    ):
        _reg_mod._ensure_entry_points()


def test_entry_point_non_callable_handler_warns():
    mock_entry_point = MagicMock()
    mock_entry_point.name = 'not_callable_style'
    mock_entry_point.value = 'my_package:not_callable'
    mock_entry_point.load.return_value = 'not_a_callable'

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.interactor_style_registry.entry_points',
            return_value=[mock_entry_point],
        ),
        pytest.warns(UserWarning, match='handler is not callable'),
    ):
        _reg_mod._ensure_entry_points()

    assert not _reg_mod._has_interactor_style('not_callable_style')


def test_entry_point_duplicate_provider_warns():
    ep1 = MagicMock()
    ep1.name = 'dup_style'
    ep1.value = 'package_a:style'
    ep1.load.return_value = _mock_style_factory

    ep2 = MagicMock()
    ep2.name = 'dup_style'
    ep2.value = 'package_b:style'
    ep2.load.return_value = _mock_style_factory

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.interactor_style_registry.entry_points',
            return_value=[ep1, ep2],
        ),
        pytest.warns(UserWarning, match='Multiple .* providers'),
    ):
        _reg_mod._ensure_entry_points()


def test_entry_point_builtin_collision_warns():
    mock_entry_point = MagicMock()
    mock_entry_point.name = 'terrain_style'
    mock_entry_point.value = 'my_package:terrain'
    mock_entry_point.load.return_value = _mock_style_factory

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.interactor_style_registry.entry_points',
            return_value=[mock_entry_point],
        ),
        pytest.warns(UserWarning, match='collides with a built-in'),
    ):
        _reg_mod._ensure_entry_points()


def test_entry_point_empty_name_warns():
    mock_entry_point = MagicMock()
    mock_entry_point.name = ''
    mock_entry_point.value = 'my_package:empty_name'
    mock_entry_point.load.return_value = _mock_style_factory

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.interactor_style_registry.entry_points',
            return_value=[mock_entry_point],
        ),
        pytest.warns(UserWarning, match='must not be empty'),
    ):
        _reg_mod._ensure_entry_points()


def test_entry_point_mapping_non_string_key_warns():
    mock_entry_point = MagicMock()
    mock_entry_point.name = 'bad_mapping'
    mock_entry_point.value = 'my_package:styles'
    mock_entry_point.load.return_value = {
        123: _mock_style_factory,
    }

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.interactor_style_registry.entry_points',
            return_value=[mock_entry_point],
        ),
        pytest.warns(UserWarning, match='style names must be strings'),
    ):
        _reg_mod._ensure_entry_points()
