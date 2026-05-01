"""Tests for the plotter component registry."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from unittest.mock import MagicMock
import warnings

import pytest

import pyvista as pv
from pyvista.plotting import component_registry as _reg_mod
from pyvista.plotting.component_registry import _CachedComponent


def _load_module_copy(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_importer(name, body):
    compiled = compile(body, f'<fake {name}>', 'exec')

    def _import(module_path):
        assert module_path == name
        module = types.ModuleType(name)
        module.__file__ = f'<fake {name}>'
        sys.modules[name] = module
        exec(compiled, module.__dict__)  # noqa: S102
        return module

    return _import


def _reset_entry_point_state(monkeypatch, eps):
    monkeypatch.setattr(_reg_mod, '_entry_points_loaded', False)
    _reg_mod._pending_components.clear()
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.entry_points',
        lambda **_: eps,
    )


def test_initial_module_state():
    """A freshly-imported registry has no registrations or pending plugins.

    Loaded as a fresh copy so the assertion is immune to any first-party
    or plugin registration that has populated the live module.
    """
    module = _load_module_copy('component_registry_test_copy', _reg_mod.__file__)
    assert module._registrations == []
    assert module._prior_values == {}
    assert module._entry_points_loaded is False
    assert module._pending_components == {}


def test_decorator_registers_component():
    @pv.register_plotter_component('demo')
    class DemoComponent:
        def __init__(self, plotter):
            self._plotter = plotter

        def title(self):
            return self._plotter.title

    pl = pv.Plotter()
    assert pl.demo.title() == pl.title


def test_register_against_subclass_target():
    """Registering against a custom subclass attaches only there."""

    class CustomTarget(pv.Plotter):
        pass

    @pv.register_plotter_component('subclass_only', target_cls=CustomTarget)
    class SubclassOnly:
        def __init__(self, plotter):
            self._plotter = plotter

    assert isinstance(CustomTarget().subclass_only, SubclassOnly)
    # Plain ``pv.Plotter`` is unaffected.
    assert 'subclass_only' not in pv.Plotter.__dict__
    assert 'subclass_only' not in pv.BasePlotter.__dict__


def test_component_init_called_once_per_instance():
    init_calls = []

    @pv.register_plotter_component('counted')
    class CountedComponent:
        def __init__(self, plotter):
            init_calls.append(plotter)
            self._plotter = plotter

    pl = pv.Plotter()
    _ = pl.counted
    _ = pl.counted
    _ = pl.counted
    assert len(init_calls) == 1
    assert init_calls[0] is pl


def test_same_cached_component_returned_on_repeat_access():
    @pv.register_plotter_component('cached')
    class CachedComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    pl = pv.Plotter()
    assert pl.cached is pl.cached


def test_class_access_returns_component_class():
    @pv.register_plotter_component('exposed')
    class ExposedComponent:
        """Visible on help."""

        def __init__(self, plotter):
            self._plotter = plotter

    assert pv.BasePlotter.exposed is ExposedComponent


def test_component_name_appears_in_dir():
    @pv.register_plotter_component('discoverable')
    class DiscoverableComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    assert 'discoverable' in dir(pv.BasePlotter)
    assert 'discoverable' in dir(pv.Plotter())


def test_distinct_plotter_instances_get_distinct_components():
    @pv.register_plotter_component('per_instance')
    class PerInstanceComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    pl1 = pv.Plotter()
    pl2 = pv.Plotter()
    assert pl1.per_instance is not pl2.per_instance


def test_close_hook_runs_on_constructed_components():
    @pv.register_plotter_component('lc_close')
    class CloseComponent:
        def __init__(self, plotter):
            self._plotter = plotter
            self.closed = False

        def __plotter_close__(self):
            self.closed = True

    pl = pv.Plotter()
    component = pl.lc_close
    pl.close()
    assert component.closed is True


def test_close_hook_skipped_for_untouched_components():
    """Never-accessed components do not construct, so their hooks never fire."""
    constructions = []

    @pv.register_plotter_component('lc_untouched')
    class UntouchedComponent:
        def __init__(self, plotter):
            constructions.append('constructed')
            self._plotter = plotter

        def __plotter_close__(self):
            constructions.append('closed')

    pl = pv.Plotter()
    pl.close()
    assert constructions == []


def test_close_hook_runs_in_reverse_construction_order():
    order = []

    @pv.register_plotter_component('lc_first')
    class FirstComponent:
        def __init__(self, plotter):
            self._plotter = plotter

        def __plotter_close__(self):
            order.append('first')

    @pv.register_plotter_component('lc_second')
    class SecondComponent:
        def __init__(self, plotter):
            self._plotter = plotter

        def __plotter_close__(self):
            order.append('second')

    pl = pv.Plotter()
    _ = pl.lc_first
    _ = pl.lc_second
    pl.close()
    assert order == ['second', 'first']


def test_components_without_close_hook_are_skipped():
    """A component with no ``__plotter_close__`` does not crash close()."""

    @pv.register_plotter_component('lc_no_hook')
    class NoHookComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    pl = pv.Plotter()
    _ = pl.lc_no_hook
    pl.close()


def test_deep_clean_hook_runs_in_reverse_order():
    order = []

    @pv.register_plotter_component('dc_a')
    class CompA:
        def __init__(self, plotter):
            self._plotter = plotter

        def __plotter_deep_clean__(self):
            order.append('a')

    @pv.register_plotter_component('dc_b')
    class CompB:
        def __init__(self, plotter):
            self._plotter = plotter

        def __plotter_deep_clean__(self):
            order.append('b')

    pl = pv.Plotter()
    _ = pl.dc_a
    _ = pl.dc_b
    pl.deep_clean()
    assert order == ['b', 'a']


def test_deep_clean_does_not_call_close_hook():
    calls = []

    @pv.register_plotter_component('split_hooks')
    class SplitComponent:
        def __init__(self, plotter):
            self._plotter = plotter

        def __plotter_close__(self):
            calls.append('close')

        def __plotter_deep_clean__(self):
            calls.append('deep')

    pl = pv.Plotter()
    _ = pl.split_hooks
    pl.deep_clean()
    assert calls == ['deep']
    pl.close()
    assert calls == ['deep', 'close']


def test_component_vs_component_collision_warns_and_replaces():
    @pv.register_plotter_component('clashing')
    class FirstComponent:
        def __init__(self, plotter):
            self._plotter = plotter

        def who(self):
            return 'first'

    with pytest.warns(UserWarning, match='replaces an existing registered'):

        @pv.register_plotter_component('clashing')
        class SecondComponent:
            def __init__(self, plotter):
                self._plotter = plotter

            def who(self):
                return 'second'

    assert pv.Plotter().clashing.who() == 'second'


def test_override_silences_replacement_warning():
    @pv.register_plotter_component('silent_override')
    class FirstComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    with warnings.catch_warnings():
        warnings.simplefilter('error')

        @pv.register_plotter_component('silent_override', override=True)
        class SecondComponent:
            def __init__(self, plotter):
                self._plotter = plotter


def test_builtin_shadow_raises_without_override():
    """``close`` is a built-in method on BasePlotter and must not be shadowed."""
    with pytest.raises(ValueError, match='shadows built-in attribute'):

        @pv.register_plotter_component('close')
        class CloseShadow:
            def __init__(self, plotter):
                self._plotter = plotter


def test_builtin_shadow_succeeds_with_override():
    """``override=True`` bypasses the built-in-shadow check and is reversible."""
    original_close = pv.BasePlotter.close

    @pv.register_plotter_component('close', override=True)
    class CloseOverride:
        def __init__(self, plotter):
            self._plotter = plotter

    try:
        # Class-level access on a _CachedComponent returns the component class.
        assert pv.BasePlotter.close is CloseOverride
    finally:
        pv.unregister_plotter_component('close')
    assert pv.BasePlotter.close is original_close


def test_empty_name_raises():
    with pytest.raises(ValueError, match='must not be empty'):
        pv.register_plotter_component('')


def test_whitespace_name_raises():
    with pytest.raises(ValueError, match='must not be empty'):
        pv.register_plotter_component('   ')


def test_non_identifier_name_raises():
    with pytest.raises(ValueError, match='must be a valid Python identifier'):
        pv.register_plotter_component('my-comp')


def test_leading_underscore_name_raises():
    with pytest.raises(ValueError, match='must not start with an underscore'):
        pv.register_plotter_component('_hidden')


def test_non_string_name_raises():
    with pytest.raises(TypeError, match='must be a string'):
        pv.register_plotter_component(123)  # type: ignore[arg-type]


def test_non_class_target_raises():
    with pytest.raises(TypeError, match='must be a class'):
        pv.register_plotter_component('name', 'not_a_class')  # type: ignore[arg-type]


def test_non_class_component_raises():
    decorator = pv.register_plotter_component('functional')
    with pytest.raises(TypeError, match='Component must be a class'):
        decorator(lambda plotter: plotter)  # type: ignore[arg-type]


def test_unregister_removes_descriptor():
    @pv.register_plotter_component('removable')
    class RemovableComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    assert 'removable' in pv.BasePlotter.__dict__
    pv.unregister_plotter_component('removable')
    assert 'removable' not in pv.BasePlotter.__dict__


def test_unregister_missing_raises():
    with pytest.raises(ValueError, match='No registered plotter component'):
        pv.unregister_plotter_component('never_registered')


def test_unregister_restores_overridden_builtin():
    """Overriding an attribute defined directly on the target is reversible."""

    class Target:
        def existing_method(self):
            return 'original'

    original_method = Target.__dict__['existing_method']

    @pv.register_plotter_component('existing_method', target_cls=Target, override=True)
    class OverrideComponent:
        def __init__(self, obj):
            self._obj = obj

    assert Target.__dict__['existing_method'] is not original_method
    pv.unregister_plotter_component('existing_method', target_cls=Target)
    assert Target.__dict__['existing_method'] is original_method


def test_registered_components_reports_records():
    @pv.register_plotter_component('introspect_me')
    class IntrospectMeComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    matches = [r for r in pv.registered_plotter_components() if r.name == 'introspect_me']
    assert len(matches) == 1
    record = matches[0]
    assert record.target is pv.BasePlotter
    assert record.component is IntrospectMeComponent
    assert record.source.endswith('IntrospectMeComponent')


def test_registered_components_returns_tuple():
    assert isinstance(pv.registered_plotter_components(), tuple)


def test_re_register_replaces_single_record():
    @pv.register_plotter_component('single_record')
    class FirstComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    with warnings.catch_warnings():
        # Suppress the intentional collision warning; the test verifies
        # the registry record state, not the warning itself.
        warnings.simplefilter('ignore')

        @pv.register_plotter_component('single_record')
        class SecondComponent:
            def __init__(self, plotter):
                self._plotter = plotter

    records = [r for r in pv.registered_plotter_components() if r.name == 'single_record']
    assert len(records) == 1
    assert records[0].component is SecondComponent


def test_save_restore_round_trip_baseline():
    before = pv.registered_plotter_components()
    state = _reg_mod._save_registry_state()
    _reg_mod._restore_registry_state(state)
    assert pv.registered_plotter_components() == before


def test_save_restore_round_trip_populated():
    @pv.register_plotter_component('snapshot_me')
    class SnapshotComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    state = _reg_mod._save_registry_state()

    @pv.register_plotter_component('post_snapshot')
    class PostSnapshotComponent:
        def __init__(self, plotter):
            self._plotter = plotter

    pv.unregister_plotter_component('snapshot_me')
    _reg_mod._restore_registry_state(state)

    names = {r.name for r in pv.registered_plotter_components()}
    assert 'snapshot_me' in names
    assert 'post_snapshot' not in names


def test_fixture_isolation_setup():
    """Register a component; the conftest fixture must clean it up."""

    @pv.register_plotter_component('leaked_component')
    class LeakedComponent:
        def __init__(self, plotter):
            self._plotter = plotter


def test_fixture_isolation_teardown():
    """The component from the previous test must not leak across tests."""
    assert 'leaked_component' not in {r.name for r in pv.registered_plotter_components()}


def test_protocol_matches_valid_component():
    class Good:
        def __init__(self, plotter):
            self._plotter = plotter

    assert isinstance(Good(None), pv.PlotterComponent)


def test_ensure_entry_points_does_not_import_plugin_modules(monkeypatch):
    """Metadata scan must not import any plugin module."""
    ep = MagicMock()
    ep.name = 'metadata_only'
    ep.value = 'metadata_only_module'

    _reset_entry_point_state(monkeypatch, [ep])
    import_calls = []
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        lambda path: import_calls.append(path) or None,
    )

    _reg_mod._ensure_entry_points()
    assert import_calls == []
    assert _reg_mod._pending_components == {'metadata_only': 'metadata_only_module'}


def test_attribute_access_triggers_plugin_load(monkeypatch):
    plugin_name = 'fake_pc_plugin_trigger'
    fake_import = _fake_importer(
        plugin_name,
        'import pyvista as pv\n'
        "@pv.register_plotter_component('ep_demo')\n"
        'class EpDemoComponent:\n'
        '    def __init__(self, plotter):\n'
        '        self._plotter = plotter\n'
        '    def value(self):\n'
        '        return 42\n',
    )
    ep = MagicMock()
    ep.name = 'ep_demo'
    ep.value = plugin_name

    _reset_entry_point_state(monkeypatch, [ep])
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        fake_import,
    )

    pl = pv.Plotter()
    try:
        assert pl.ep_demo.value() == 42
        # Cached on second access.
        assert pl.ep_demo.value() == 42
    finally:
        with contextlib.suppress(ValueError):
            pv.unregister_plotter_component('ep_demo')
        sys.modules.pop(plugin_name, None)


def test_unrelated_attribute_does_not_load_plugin(monkeypatch):
    """An attribute not matching any pending plugin must not import any plugin."""
    ep = MagicMock()
    ep.name = 'should_not_load'
    ep.value = 'should_not_load_module'

    _reset_entry_point_state(monkeypatch, [ep])
    import_mock = MagicMock()
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        import_mock,
    )

    pl = pv.Plotter()
    with pytest.raises(AttributeError):
        _ = pl.definitely_not_a_plugin_name
    import_mock.assert_not_called()


def test_pending_plugin_only_imported_once(monkeypatch):
    plugin_name = 'fake_pc_one_shot'
    import_count = 0

    def _counting_import(path):
        nonlocal import_count
        import_count += 1
        module = types.ModuleType(path)
        sys.modules[path] = module
        exec(  # noqa: S102
            'import pyvista as pv\n'
            "@pv.register_plotter_component('one_shot')\n"
            'class OneShotComponent:\n'
            '    def __init__(self, plotter):\n'
            '        self._plotter = plotter\n',
            module.__dict__,
        )
        return module

    ep = MagicMock()
    ep.name = 'one_shot'
    ep.value = plugin_name

    _reset_entry_point_state(monkeypatch, [ep])
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        _counting_import,
    )

    plotters = [pv.Plotter() for _ in range(3)]
    try:
        for pl in plotters:
            _ = pl.one_shot
        assert import_count == 1
    finally:
        with contextlib.suppress(ValueError):
            pv.unregister_plotter_component('one_shot')
        sys.modules.pop(plugin_name, None)


def test_broken_plugin_warns_once_and_isolates(monkeypatch):
    """A failing plugin import warns once and is dropped from the pending list."""
    ep = MagicMock()
    ep.name = 'broken'
    ep.value = 'broken_pc_module'

    _reset_entry_point_state(monkeypatch, [ep])
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        MagicMock(side_effect=ImportError('missing dep')),
    )

    pl = pv.Plotter()
    with pytest.warns(UserWarning, match='Failed to load'):
        with pytest.raises(AttributeError):
            _ = pl.broken

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        with pytest.raises(AttributeError):
            _ = pl.broken
    component_warnings = [w for w in captured if 'Failed to load' in str(w.message)]
    assert component_warnings == []


def test_decorator_wins_over_pending_entry_point(monkeypatch):
    """A decorator-attached component preempts a pending entry-point plugin."""

    @pv.register_plotter_component('ep_preempted')
    class DecoratorComponent:
        def __init__(self, plotter):
            self._plotter = plotter

        def who(self):
            return 'decorator'

    ep = MagicMock()
    ep.name = 'ep_preempted'
    ep.value = 'fake_pc_preempted_module'

    _reset_entry_point_state(monkeypatch, [ep])
    import_mock = MagicMock()
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        import_mock,
    )

    assert pv.Plotter().ep_preempted.who() == 'decorator'
    import_mock.assert_not_called()


def test_registered_components_forces_discovery(monkeypatch):
    """``registered_plotter_components()`` forces discovery of pending plugins."""
    plugin_name = 'fake_pc_plugin_forced'
    fake_import = _fake_importer(
        plugin_name,
        'import pyvista as pv\n'
        "@pv.register_plotter_component('forced')\n"
        'class ForcedComponent:\n'
        '    def __init__(self, plotter):\n'
        '        self._plotter = plotter\n',
    )
    ep = MagicMock()
    ep.name = 'forced'
    ep.value = plugin_name

    _reset_entry_point_state(monkeypatch, [ep])
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        fake_import,
    )

    try:
        names = {r.name for r in pv.registered_plotter_components()}
        assert 'forced' in names
    finally:
        with contextlib.suppress(ValueError):
            pv.unregister_plotter_component('forced')
        sys.modules.pop(plugin_name, None)


def test_pending_component_appears_in_dir_without_loading(monkeypatch):
    """``dir(plotter)`` lists pending plugin names without loading them."""
    plugin_name = 'fake_pc_dir'
    fake_import = _fake_importer(
        plugin_name,
        'import pyvista as pv\n'
        "@pv.register_plotter_component('dir_demo')\n"
        'class DirDemoComponent:\n'
        '    def __init__(self, plotter):\n'
        '        self._plotter = plotter\n',
    )
    ep = MagicMock()
    ep.name = 'dir_demo'
    ep.value = plugin_name

    _reset_entry_point_state(monkeypatch, [ep])
    import_calls = []

    def _tracking_import(module_path):
        import_calls.append(module_path)
        return fake_import(module_path)

    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        _tracking_import,
    )

    listing = dir(pv.Plotter())
    assert 'dir_demo' in listing
    assert import_calls == []
    sys.modules.pop(plugin_name, None)


def test_explicit_component_appears_in_dir():
    @pv.register_plotter_component('explicit_dir_demo')
    class ExplicitDirDemo:
        def __init__(self, plotter):
            self._plotter = plotter

    assert 'explicit_dir_demo' in dir(pv.Plotter())


def test_dir_after_pending_component_resolved(monkeypatch):
    """Once a pending plugin resolves, ``dir`` finds it via the descriptor."""
    plugin_name = 'fake_pc_dir_resolved'
    fake_import = _fake_importer(
        plugin_name,
        'import pyvista as pv\n'
        "@pv.register_plotter_component('resolved_demo')\n"
        'class ResolvedDemoComponent:\n'
        '    def __init__(self, plotter):\n'
        '        self._plotter = plotter\n',
    )
    ep = MagicMock()
    ep.name = 'resolved_demo'
    ep.value = plugin_name

    _reset_entry_point_state(monkeypatch, [ep])
    monkeypatch.setattr(
        'pyvista.plotting.component_registry.import_module',
        fake_import,
    )

    pl = pv.Plotter()
    try:
        _ = pl.resolved_demo
        assert _reg_mod._pending_components == {}
        assert 'resolved_demo' in dir(pl)
    finally:
        sys.modules.pop(plugin_name, None)
        if 'resolved_demo' in pv.BasePlotter.__dict__:
            delattr(pv.BasePlotter, 'resolved_demo')


def test_no_entry_points_is_silent(monkeypatch):
    """No installed component plugins must be a no-op without warnings."""
    _reset_entry_point_state(monkeypatch, [])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        _reg_mod._ensure_entry_points()
        with pytest.raises(AttributeError):
            _ = pv.Plotter().completely_unknown

    component_warnings = [
        w
        for w in captured
        if 'pyvista.plotter_components' in str(w.message) or 'Failed to load' in str(w.message)
    ]
    assert component_warnings == []


def test_freeze_still_rejects_arbitrary_attribute_writes():
    """Constructing a component must not weaken the no-new-attributes freeze."""

    @pv.register_plotter_component('freeze_probe')
    class FreezeProbe:
        def __init__(self, plotter):
            self._plotter = plotter

    pl = pv.Plotter()
    _ = pl.freeze_probe
    with pytest.raises(pv.PyVistaAttributeError):
        pl.this_attribute_should_not_exist = 1


def test_plotter_subclass_inherits_components():
    """A subclass of Plotter (mimicking pyvistaqt.QtInteractor) inherits components."""

    @pv.register_plotter_component('subclass_demo')
    class SubclassDemo:
        def __init__(self, plotter):
            self._plotter = plotter

    class CustomPlotter(pv.Plotter):
        pass

    assert isinstance(CustomPlotter().subclass_demo, SubclassDemo)


def test_scalar_bars_is_a_registered_component():
    names = {r.name for r in pv.registered_plotter_components()}
    assert 'scalar_bars' in names


def test_scalar_bars_lazily_constructed_and_cached():
    pl = pv.Plotter()
    # Before access, the instance dict has no cached entry.
    assert 'scalar_bars' not in pl.__dict__
    sb = pl.scalar_bars
    # After access, the cache entry is present and the component is tracked
    # for close-time iteration.
    assert pl.__dict__['scalar_bars'] is sb
    assert sb in pl._components


def test_scalar_bars_close_hook_clears_state():
    sphere = pv.Sphere()
    sphere['Data'] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere)
    sb = pl.scalar_bars
    assert len(sb._scalar_bar_actors) > 0
    pl.close()
    assert sb._scalar_bar_actors == {}


def test_scalar_bars_class_level_access_returns_class():
    from pyvista.plotting.scalar_bars import ScalarBars

    assert pv.BasePlotter.scalar_bars is ScalarBars


def test_inherited_component_replacement_warns():
    """Re-registering a component on a subclass when parent owns it warns about inheritance."""

    class ParentTarget:
        pass

    class ChildTarget(ParentTarget):
        pass

    @pv.register_plotter_component('inherited_demo', target_cls=ParentTarget)
    class FirstComponent:
        def __init__(self, obj):
            self._obj = obj

    with pytest.warns(UserWarning, match='inherited by'):

        @pv.register_plotter_component('inherited_demo', target_cls=ChildTarget)
        class SecondComponent:
            def __init__(self, obj):
                self._obj = obj


def test_inherited_shadow_raises_with_inherited_location():
    """Registering on a subclass when parent shadows the name reports parent location."""

    class ParentTarget:
        def existing_method(self):
            return 'parent'

    class ChildTarget(ParentTarget):
        pass

    with pytest.raises(ValueError, match='inherited by'):

        @pv.register_plotter_component('existing_method', target_cls=ChildTarget)
        class ShadowComponent:
            def __init__(self, obj):
                self._obj = obj


def test_component_descriptor_handles_slots_target():
    """``__slots__`` targets without ``__dict__`` get a fresh component each access."""

    class Component:
        def __init__(self, obj):
            self._obj = obj

    class SlotsTarget:
        __slots__ = ()

    SlotsTarget.demo_slots = _CachedComponent('demo_slots', Component)
    obj = SlotsTarget()
    first = obj.demo_slots
    second = obj.demo_slots
    # ``__dict__`` is unavailable: each access yields a new instance and
    # nothing is tracked.
    assert isinstance(first, Component)
    assert isinstance(second, Component)
    assert first is not second


def test_component_descriptor_rejects_lifecycle_hooks_on_slots_target():
    """Lifecycle hooks on a slots target raise rather than silently no-op.

    A component that declares ``__plotter_close__`` or
    ``__plotter_deep_clean__`` cannot be attached to a target without
    a ``__dict__`` because each access constructs a fresh instance —
    there is no stable receiver for close-time teardown, so any
    cleanup the hook is responsible for would silently leak.
    """

    class CloseHookComponent:
        def __init__(self, obj):
            self._obj = obj

        def __plotter_close__(self):  # pragma: no cover — never invoked
            pass

    class DeepCleanHookComponent:
        def __init__(self, obj):
            self._obj = obj

        def __plotter_deep_clean__(self):  # pragma: no cover — never invoked
            pass

    class SlotsTarget:
        __slots__ = ()

    SlotsTarget.with_close = _CachedComponent('with_close', CloseHookComponent)
    SlotsTarget.with_deep = _CachedComponent('with_deep', DeepCleanHookComponent)
    obj = SlotsTarget()

    with pytest.raises(TypeError, match=r'__plotter_close__'):
        _ = obj.with_close
    with pytest.raises(TypeError, match=r'__plotter_deep_clean__'):
        _ = obj.with_deep


def test_component_descriptor_slots_error_names_target_and_hooks():
    """The slots-rejection error identifies the offending names for debugging."""

    class HookyComponent:
        def __init__(self, obj):
            self._obj = obj

        def __plotter_close__(self):  # pragma: no cover — never invoked
            pass

        def __plotter_deep_clean__(self):  # pragma: no cover — never invoked
            pass

    class FrozenPlotterTarget:
        __slots__ = ()

    FrozenPlotterTarget.hooky = _CachedComponent('hooky', HookyComponent)
    obj = FrozenPlotterTarget()

    with pytest.raises(TypeError) as excinfo:
        _ = obj.hooky
    message = str(excinfo.value)
    assert "'hooky'" in message
    assert 'HookyComponent' in message
    assert 'FrozenPlotterTarget' in message
    assert '__plotter_close__' in message
    assert '__plotter_deep_clean__' in message
    assert '__slots__' in message


def test_component_descriptor_slots_target_no_lifecycle_still_works():
    """A hookless component on a slots target still falls through cleanly.

    The lifecycle-hook rejection is targeted: components without hooks
    keep the previous "fresh instance per access" behavior so existing
    slots-based extension points are unaffected.
    """

    class HooklessComponent:
        def __init__(self, obj):
            self._obj = obj

    class SlotsTarget:
        __slots__ = ()

    SlotsTarget.hookless = _CachedComponent('hookless', HooklessComponent)
    obj = SlotsTarget()

    first = obj.hookless
    second = obj.hookless
    assert isinstance(first, HooklessComponent)
    assert first is not second
