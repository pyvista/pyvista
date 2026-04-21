"""Tests for theme registration and discovery."""

from __future__ import annotations

import os
from typing import ClassVar
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import pytest

import pyvista as pv
from pyvista.plotting import theme_registry as _reg_mod
from pyvista.plotting.themes import DarkTheme
from pyvista.plotting.themes import DocumentTheme
from pyvista.plotting.themes import ParaViewTheme
from pyvista.plotting.themes import Theme
from pyvista.plotting.themes import _set_plot_theme_from_env


def test_init_subclass_auto_registers():
    class FooTheme(Theme):
        _default_name: ClassVar[str] = 'foo_theme'

    assert _reg_mod._resolve_theme('foo_theme') is not None
    assert 'foo_theme' in _reg_mod._available_theme_names()

    pv.set_plot_theme('foo_theme')
    assert pv.global_theme.name == 'foo_theme'


def test_init_subclass_is_case_insensitive():
    class CaseTheme(Theme):
        _default_name: ClassVar[str] = 'MyCase'

    assert _reg_mod._resolve_theme('mycase') is not None
    assert _reg_mod._resolve_theme('MYCASE') is not None


def test_init_subclass_subclass_of_subclass_registers_own_name():
    class Derived(DocumentTheme):
        _default_name: ClassVar[str] = 'derived_doc'

    # Derived registers as 'derived_doc', not 'document'
    assert _reg_mod._resolve_theme('derived_doc') is not None
    resolved = _reg_mod._resolve_theme('derived_doc')
    assert isinstance(resolved, Derived)

    # DocumentTheme still resolves to DocumentTheme, not Derived
    doc = _reg_mod._resolve_theme('document')
    assert type(doc) is DocumentTheme


def test_init_subclass_missing_default_name_is_silent():
    """Ad-hoc subclasses without ``_default_name`` should register silently."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')

        class NoName(Theme):
            pass

    assert 'noname' not in _reg_mod._available_theme_names()


def test_init_subclass_empty_default_name_warns():
    with pytest.warns(UserWarning, match="invalid '_default_name'"):

        class EmptyName(Theme):
            _default_name: ClassVar[str] = ''


def test_init_subclass_whitespace_only_name_warns():
    with pytest.warns(UserWarning, match='empty ``_default_name``'):

        class WhitespaceName(Theme):
            _default_name: ClassVar[str] = '   '


def test_init_subclass_duplicate_name_warns():
    class OriginalTheme(Theme):
        _default_name: ClassVar[str] = 'dup_name'

    with pytest.warns(UserWarning, match='is already registered'):

        class DupTheme(Theme):
            _default_name: ClassVar[str] = 'dup_name'

    # First registration wins
    resolved = _reg_mod._resolve_theme('dup_name')
    assert isinstance(resolved, OriginalTheme)


def test_register_theme_instance():
    custom = DarkTheme()
    custom.show_edges = True

    returned = pv.register_theme('my_custom', custom)
    assert returned is custom

    pv.set_plot_theme('my_custom')
    assert pv.global_theme.show_edges is True


def test_register_theme_override():
    first = DarkTheme()
    pv.register_theme('replaceable', first)

    replacement = ParaViewTheme()
    pv.register_theme('replaceable', replacement, override=True)

    resolved = _reg_mod._resolve_theme('replaceable')
    assert isinstance(resolved, ParaViewTheme)


def test_register_theme_collision_raises():
    instance = DarkTheme()
    pv.register_theme('collision_name', instance)

    with pytest.raises(ValueError, match='already registered'):
        pv.register_theme('collision_name', DocumentTheme())


def test_register_theme_rejects_class():
    with pytest.raises(TypeError, match='expects a Theme instance, got class'):
        pv.register_theme('as_class', DarkTheme)


def test_register_theme_rejects_non_theme():
    with pytest.raises(TypeError, match='expects a pyvista Theme instance'):
        pv.register_theme('bogus', 'not a theme')


def test_register_theme_empty_name_raises():
    with pytest.raises(ValueError, match='must not be empty'):
        pv.register_theme('', DarkTheme())


def test_register_theme_collides_with_subclass_registration():
    class PreExisting(Theme):
        _default_name: ClassVar[str] = 'pre_existing'

    with pytest.raises(ValueError, match='already registered'):
        pv.register_theme('pre_existing', DarkTheme())


def test_dotted_path_resolution():
    pv.set_plot_theme('pyvista.plotting.themes:DarkTheme')
    assert pv.global_theme.name == 'dark'


def test_dotted_path_wrong_type_raises():
    with pytest.raises(ValueError, match='does not resolve to a pyvista Theme'):
        pv.set_plot_theme('pyvista.plotting.themes:Color')


def test_dotted_path_missing_module_raises():
    with pytest.raises(ValueError, match='Cannot import'):
        pv.set_plot_theme('no_such_module_xyz:Anything')


def test_dotted_path_missing_attribute_raises():
    with pytest.raises(ValueError, match='not found in module'):
        pv.set_plot_theme('pyvista.plotting.themes:NoSuchTheme')


def test_dotted_path_empty_parts_raises():
    with pytest.raises(ValueError, match='Invalid theme spec'):
        pv.set_plot_theme(':Bare')
    with pytest.raises(ValueError, match='Invalid theme spec'):
        pv.set_plot_theme('module:')


def test_unknown_name_raises():
    with pytest.raises(ValueError, match='not found'):
        pv.set_plot_theme('definitely_not_a_theme')


def test_entry_point_subclass_registered_via_init_subclass_resolves():
    """An entry-point module whose class self-registers via ``__init_subclass__``
    must still resolve by name when the registry is otherwise empty."""

    class SelfRegistering(Theme):
        _default_name: ClassVar[str] = 'self_registering'

    # Clear the explicit registration so we exercise the entry-point path.
    _reg_mod._registered_theme_classes.pop('self_registering', None)
    _reg_mod._registered_theme_classes_sources.pop('self_registering', None)

    # Simulate an entry point whose load() re-imports the module and
    # re-triggers __init_subclass__ on SelfRegistering.
    def fake_load():
        _reg_mod._register_theme_class(
            'self_registering',
            SelfRegistering,
            source='fake',
        )
        return SelfRegistering

    ep = MagicMock()
    ep.name = 'self_registering'
    ep.value = 'pkg:SelfRegistering'
    ep.load.side_effect = fake_load

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.theme_registry.entry_points',
        return_value=[ep],
    ):
        resolved = _reg_mod._resolve_theme('self_registering')

    assert isinstance(resolved, SelfRegistering)


def test_entry_point_class_discovery():
    class EpTheme(Theme):
        _default_name: ClassVar[str] = 'ep_registered'

    # Clear explicit registration so discovery path is exercised
    _reg_mod._registered_theme_classes.pop('ep_registered', None)
    _reg_mod._registered_theme_classes_sources.pop('ep_registered', None)

    ep = MagicMock()
    ep.name = 'ep_registered'
    ep.value = 'pkg:EpTheme'
    ep.load.return_value = EpTheme

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.theme_registry.entry_points',
        return_value=[ep],
    ):
        resolved = _reg_mod._resolve_theme('ep_registered')

    assert isinstance(resolved, EpTheme)


def test_entry_point_instance_discovery():
    ep_theme_instance = DarkTheme()
    ep_theme_instance.show_edges = True

    ep = MagicMock()
    ep.name = 'ep_instance'
    ep.value = 'pkg:theme_instance'
    ep.load.return_value = ep_theme_instance

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.theme_registry.entry_points',
        return_value=[ep],
    ):
        resolved = _reg_mod._resolve_theme('ep_instance')

    assert resolved is ep_theme_instance


def test_entry_point_mapping_discovery():
    class MapATheme(Theme):
        _default_name: ClassVar[str] = 'map_a'

    class MapBTheme(Theme):
        _default_name: ClassVar[str] = 'map_b'

    # Clear explicit registrations so the entry-point mapping path is used
    for key in ('map_a', 'map_b'):
        _reg_mod._registered_theme_classes.pop(key, None)
        _reg_mod._registered_theme_classes_sources.pop(key, None)

    ep = MagicMock()
    ep.name = 'bulk'
    ep.value = 'pkg:THEMES'
    ep.load.return_value = {'map_a': MapATheme, 'map_b': MapBTheme}

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.theme_registry.entry_points',
        return_value=[ep],
    ):
        assert isinstance(_reg_mod._resolve_theme('map_a'), MapATheme)
        assert isinstance(_reg_mod._resolve_theme('map_b'), MapBTheme)


def test_entry_point_discovery_is_cached():
    class CachedTheme(Theme):
        _default_name: ClassVar[str] = 'cached_theme'

    _reg_mod._registered_theme_classes.pop('cached_theme', None)
    _reg_mod._registered_theme_classes_sources.pop('cached_theme', None)

    ep = MagicMock()
    ep.name = 'cached_theme'
    ep.value = 'pkg:CachedTheme'
    ep.load.return_value = CachedTheme

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.theme_registry.entry_points',
        return_value=[ep],
    ):
        _reg_mod._resolve_theme('cached_theme')
        ep.load.assert_called_once()

        _reg_mod._resolve_theme('cached_theme')
        ep.load.assert_called_once()


def test_entry_point_explicit_wins_silently():
    class ExplicitTheme(Theme):
        _default_name: ClassVar[str] = 'explicit_wins'

    # EntryPointTheme is defined under a distinct name to avoid colliding
    # with ExplicitTheme during class creation. The entry point below still
    # advertises it under 'explicit_wins' — the explicit registration should
    # win.
    class EntryPointTheme(Theme):
        _default_name: ClassVar[str] = '__ep_source__'

    ep = MagicMock()
    ep.name = 'explicit_wins'
    ep.value = 'pkg:EntryPointTheme'
    ep.load.return_value = EntryPointTheme

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.theme_registry.entry_points',
        return_value=[ep],
    ):
        resolved = _reg_mod._resolve_theme('explicit_wins')

    assert isinstance(resolved, ExplicitTheme)


def test_entry_point_load_failure_warns():
    ep = MagicMock()
    ep.name = 'broken_theme'
    ep.value = 'pkg:broken'
    ep.load.side_effect = ImportError('missing dependency')

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.theme_registry.entry_points',
            return_value=[ep],
        ),
        pytest.warns(UserWarning, match='Failed to load'),
    ):
        _reg_mod._ensure_entry_points()


def test_entry_point_non_theme_warns():
    ep = MagicMock()
    ep.name = 'not_theme'
    ep.value = 'pkg:not_theme'
    ep.load.return_value = 'not a theme'

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.theme_registry.entry_points',
            return_value=[ep],
        ),
        pytest.warns(UserWarning, match='not a pyvista Theme'),
    ):
        _reg_mod._ensure_entry_points()


def test_entry_point_duplicate_provider_warns():
    # Distinct _default_name so the class itself registers cleanly; the
    # two entry points below both advertise it under 'dup_theme'.
    class DupTheme(Theme):
        _default_name: ClassVar[str] = '__dup_source__'

    _reg_mod._registered_theme_classes.pop('dup_theme', None)
    _reg_mod._registered_theme_classes_sources.pop('dup_theme', None)

    ep1 = MagicMock()
    ep1.name = 'dup_theme'
    ep1.value = 'pkg_a:theme'
    ep1.load.return_value = DupTheme

    ep2 = MagicMock()
    ep2.name = 'dup_theme'
    ep2.value = 'pkg_b:theme'
    ep2.load.return_value = DupTheme

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.theme_registry.entry_points',
            return_value=[ep1, ep2],
        ),
        pytest.warns(UserWarning, match='Multiple .* providers'),
    ):
        _reg_mod._ensure_entry_points()


def test_entry_point_mapping_non_string_key_warns():
    class MappingTheme(Theme):
        _default_name: ClassVar[str] = '__mapping_source__'

    ep = MagicMock()
    ep.name = 'bad_mapping'
    ep.value = 'pkg:THEMES'
    ep.load.return_value = {123: MappingTheme}

    _reg_mod._entry_points_loaded = False
    with (
        patch(
            'pyvista.plotting.theme_registry.entry_points',
            return_value=[ep],
        ),
        pytest.warns(UserWarning, match='theme names must be strings'),
    ):
        _reg_mod._ensure_entry_points()


def test_env_var_registered_name():
    class EnvTheme(Theme):
        _default_name: ClassVar[str] = 'env_theme'

    try:
        os.environ['PYVISTA_PLOT_THEME'] = 'env_theme'
        _set_plot_theme_from_env()
    finally:
        os.environ.pop('PYVISTA_PLOT_THEME', None)

    assert pv.global_theme.name == 'env_theme'


def test_env_var_dotted_path_preserves_case():
    try:
        os.environ['PYVISTA_PLOT_THEME'] = 'pyvista.plotting.themes:DarkTheme'
        _set_plot_theme_from_env()
    finally:
        os.environ.pop('PYVISTA_PLOT_THEME', None)

    assert pv.global_theme.name == 'dark'


def test_env_var_unknown_warns():
    try:
        os.environ['PYVISTA_PLOT_THEME'] = 'definitely_bogus'
        with pytest.warns(UserWarning, match='Invalid PYVISTA_PLOT_THEME'):
            _set_plot_theme_from_env()
    finally:
        os.environ.pop('PYVISTA_PLOT_THEME', None)


def test_available_theme_names_includes_builtins():
    names = _reg_mod._available_theme_names()
    for name in ('dark', 'document', 'paraview', 'default', 'vtk'):
        assert name in names


def test_registered_themes_includes_builtins_and_kinds():
    registered = pv.registered_themes()
    expected_classes = {
        'dark': 'DarkTheme',
        'document': 'DocumentTheme',
        'paraview': 'ParaViewTheme',
        'testing': '_TestingTheme',
    }
    for name, class_name in expected_classes.items():
        assert registered[name].kind == 'subclass'
        assert registered[name].source.endswith(class_name)
    for name in ('default', 'vtk'):
        assert registered[name].kind == 'alias'


def test_registered_themes_reports_instance_source():
    custom = DarkTheme()
    pv.register_theme('reg_inst_theme', custom)

    record = pv.registered_themes()['reg_inst_theme']
    assert record.name == 'reg_inst_theme'
    assert record.kind == 'instance'
    assert "register_theme('reg_inst_theme')" in record.source


def test_registered_themes_reports_entry_point_source():
    class EpListedTheme(Theme):
        _default_name: ClassVar[str] = '__ep_listed__'

    _reg_mod._registered_theme_classes.pop('ep_listed', None)
    _reg_mod._registered_theme_classes_sources.pop('ep_listed', None)

    ep = MagicMock()
    ep.name = 'ep_listed'
    ep.value = 'some_plugin.pkg:EpListedTheme'
    ep.load.return_value = EpListedTheme

    _reg_mod._entry_points_loaded = False
    with patch(
        'pyvista.plotting.theme_registry.entry_points',
        return_value=[ep],
    ):
        registered = pv.registered_themes()

    record = registered['ep_listed']
    assert record.kind == 'entry_point'
    assert record.source == 'some_plugin.pkg:EpListedTheme'


def test_registered_themes_is_sorted():
    registered = pv.registered_themes()
    assert list(registered) == sorted(registered)
