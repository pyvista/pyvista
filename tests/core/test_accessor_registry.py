"""Tests for the dataset accessor registry."""

from __future__ import annotations

import warnings

import pytest

import pyvista as pv
from pyvista.core.utilities import accessor_registry as _reg_mod


def test_decorator_registers_accessor():
    @pv.register_dataset_accessor('demo', pv.PolyData)
    class DemoAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

        def n_points_doubled(self):
            return self._mesh.n_points * 2

    sphere = pv.Sphere()
    assert sphere.demo.n_points_doubled() == sphere.n_points * 2


def test_direct_call_registers_accessor():
    class DemoAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

        def value(self):
            return 42

    pv.register_dataset_accessor('demo_direct', pv.PolyData)(DemoAccessor)
    assert pv.Sphere().demo_direct.value() == 42


def test_accessor_init_called_once_per_instance():
    init_calls: list[pv.PolyData] = []

    @pv.register_dataset_accessor('counted', pv.PolyData)
    class CountedAccessor:
        def __init__(self, mesh):
            init_calls.append(mesh)
            self._mesh = mesh

    sphere = pv.Sphere()
    _ = sphere.counted
    _ = sphere.counted
    _ = sphere.counted
    assert len(init_calls) == 1
    assert init_calls[0] is sphere


def test_same_cached_accessor_returned_on_repeat_access():
    @pv.register_dataset_accessor('cached', pv.PolyData)
    class CachedAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    sphere = pv.Sphere()
    first = sphere.cached
    second = sphere.cached
    assert first is second


def test_del_evicts_cache():
    """Deleting the instance attribute forces a rebuild."""
    init_count = 0

    @pv.register_dataset_accessor('evictable', pv.PolyData)
    class EvictableAccessor:
        def __init__(self, mesh):
            nonlocal init_count
            init_count += 1
            self._mesh = mesh

    sphere = pv.Sphere()
    _ = sphere.evictable
    assert init_count == 1
    del sphere.evictable
    _ = sphere.evictable
    assert init_count == 2


def test_class_access_returns_accessor_class():
    @pv.register_dataset_accessor('exposed', pv.PolyData)
    class ExposedAccessor:
        """Docstring visible on help(pv.PolyData.exposed)."""

        def __init__(self, mesh):
            self._mesh = mesh

    assert pv.PolyData.exposed is ExposedAccessor
    assert 'Docstring visible' in (pv.PolyData.exposed.__doc__ or '')


def test_accessor_name_appears_in_dir():
    @pv.register_dataset_accessor('discoverable', pv.PolyData)
    class DiscoverableAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    assert 'discoverable' in dir(pv.PolyData)
    assert 'discoverable' in dir(pv.Sphere())


@pytest.mark.parametrize(
    'factory',
    [
        pv.Sphere,
        pv.ImageData,
        pv.RectilinearGrid,
        pv.UnstructuredGrid,
        pv.StructuredGrid,
    ],
)
def test_register_on_dataset_visible_on_all_subclasses(factory):
    @pv.register_dataset_accessor('ds_level', pv.DataSet)
    class DataSetLevelAccessor:
        def __init__(self, dataset):
            self._dataset = dataset

        def is_alive(self):
            return self._dataset is not None

    obj = factory()
    assert obj.ds_level.is_alive() is True


def test_register_on_dataobject_visible_on_multiblock():
    @pv.register_dataset_accessor('obj_level', pv.DataObject)
    class DataObjectLevelAccessor:
        def __init__(self, obj):
            self._obj = obj

        def tag(self):
            return type(self._obj).__name__

    block = pv.MultiBlock()
    assert block.obj_level.tag() == 'MultiBlock'
    assert pv.Sphere().obj_level.tag() == 'PolyData'


def test_register_on_polydata_not_visible_on_unstructured():
    @pv.register_dataset_accessor('poly_only', pv.PolyData)
    class PolyOnlyAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    assert hasattr(pv.Sphere(), 'poly_only')
    assert not hasattr(pv.UnstructuredGrid(), 'poly_only')


def test_subclass_accessor_shadows_ancestor_silently_when_different_names():
    """Registering different names on ancestor and subclass does not collide."""

    @pv.register_dataset_accessor('base_ns', pv.DataSet)
    class BaseAccessor:
        def __init__(self, dataset):
            self._dataset = dataset

        def who(self):
            return 'base'

    @pv.register_dataset_accessor('sub_ns', pv.PolyData)
    class SubAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

        def who(self):
            return 'sub'

    sphere = pv.Sphere()
    assert sphere.base_ns.who() == 'base'
    assert sphere.sub_ns.who() == 'sub'

    grid = pv.ImageData()
    assert grid.base_ns.who() == 'base'
    assert not hasattr(grid, 'sub_ns')


def test_accessor_survives_type_changing_filter_chain():
    """An accessor on DataSet survives a filter that changes dataset type."""

    @pv.register_dataset_accessor('chain_tag', pv.DataSet)
    class ChainTagAccessor:
        def __init__(self, dataset):
            self._dataset = dataset

        def classname(self):
            return type(self._dataset).__name__

    sphere = pv.Sphere()
    grid = sphere.cast_to_unstructured_grid()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.chain_tag.classname() == 'UnstructuredGrid'


def test_accessor_vs_accessor_collision_warns_and_replaces():
    @pv.register_dataset_accessor('clashing', pv.PolyData)
    class FirstAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

        def who(self):
            return 'first'

    with pytest.warns(UserWarning, match='replaces an existing registered accessor'):

        @pv.register_dataset_accessor('clashing', pv.PolyData)
        class SecondAccessor:
            def __init__(self, mesh):
                self._mesh = mesh

            def who(self):
                return 'second'

    assert pv.Sphere().clashing.who() == 'second'


def test_inherited_accessor_collision_warns():
    """Registering on a subclass when an ancestor already owns the name warns."""

    @pv.register_dataset_accessor('inherited_clash', pv.DataSet)
    class AncestorAccessor:
        def __init__(self, dataset):
            self._dataset = dataset

        def who(self):
            return 'ancestor'

    with pytest.warns(UserWarning, match='inherited by PolyData'):

        @pv.register_dataset_accessor('inherited_clash', pv.PolyData)
        class DescendantAccessor:
            def __init__(self, mesh):
                self._mesh = mesh

            def who(self):
                return 'descendant'

    assert pv.Sphere().inherited_clash.who() == 'descendant'
    assert pv.ImageData().inherited_clash.who() == 'ancestor'


def test_builtin_shadow_raises_without_override():
    """Shadowing an existing filter method must raise, not warn."""
    with pytest.raises(ValueError, match='shadows built-in attribute'):

        @pv.register_dataset_accessor('clip', pv.PolyData)
        class ClipAccessor:
            def __init__(self, mesh):
                self._mesh = mesh

    assert callable(pv.PolyData.clip)


def test_builtin_shadow_succeeds_with_override():
    """``override=True`` bypasses the built-in-shadow check."""
    original_clip = pv.PolyData.clip

    @pv.register_dataset_accessor('clip', pv.PolyData, override=True)
    class FakeClipAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

        def kind(self):
            return 'accessor'

    assert pv.Sphere().clip.kind() == 'accessor'
    # Class-level access on a _CachedAccessor returns the accessor class itself.
    assert pv.PolyData.clip is FakeClipAccessor

    pv.unregister_dataset_accessor('clip', pv.PolyData)
    assert pv.PolyData.clip is original_clip
    # ``clip`` is inherited from DataSetFilters, not re-attached to PolyData.
    assert 'clip' not in pv.PolyData.__dict__


def test_inherited_builtin_shadow_raises():
    """A property defined on an ancestor should still be shadow-detected."""
    # ``bounds`` is a property inherited from DataSet.
    with pytest.raises(ValueError, match='shadows built-in attribute'):

        @pv.register_dataset_accessor('bounds', pv.PolyData)
        class BoundsAccessor:
            def __init__(self, mesh):
                self._mesh = mesh


def test_empty_name_raises():
    with pytest.raises(ValueError, match='must not be empty'):
        pv.register_dataset_accessor('', pv.PolyData)


def test_whitespace_only_name_raises():
    with pytest.raises(ValueError, match='must not be empty'):
        pv.register_dataset_accessor('   ', pv.PolyData)


def test_non_identifier_name_raises():
    with pytest.raises(ValueError, match='must be a valid Python identifier'):
        pv.register_dataset_accessor('my-accessor', pv.PolyData)


def test_leading_digit_name_raises():
    with pytest.raises(ValueError, match='must be a valid Python identifier'):
        pv.register_dataset_accessor('1foo', pv.PolyData)


def test_leading_underscore_name_raises():
    with pytest.raises(ValueError, match='must not start with an underscore'):
        pv.register_dataset_accessor('_hidden', pv.PolyData)


def test_non_string_name_raises():
    with pytest.raises(TypeError, match='must be a string'):
        # Intentional type violation: the runtime guard rejects non-strings.
        pv.register_dataset_accessor(123, pv.PolyData)  # type: ignore[arg-type]


def test_non_class_target_raises():
    with pytest.raises(TypeError, match='must be a class'):
        # Intentional type violation: the runtime guard rejects non-classes.
        pv.register_dataset_accessor('name', 'not_a_class')  # type: ignore[arg-type]


def test_non_class_accessor_raises():
    decorator = pv.register_dataset_accessor('functional', pv.PolyData)
    with pytest.raises(TypeError, match='Accessor must be a class'):
        # Intentional type violation: the decorator rejects plain callables.
        decorator(lambda mesh: mesh)  # type: ignore[arg-type]


def test_unregister_removes_descriptor():
    @pv.register_dataset_accessor('removable', pv.PolyData)
    class RemovableAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    assert 'removable' in pv.PolyData.__dict__
    pv.unregister_dataset_accessor('removable', pv.PolyData)
    assert 'removable' not in pv.PolyData.__dict__


def test_unregister_restores_overridden_builtin():
    """Overriding an attribute defined directly on the target is reversible."""

    class Target:
        def existing_method(self):
            return 'original'

    original_method = Target.__dict__['existing_method']

    @pv.register_dataset_accessor('existing_method', Target, override=True)
    class OverrideAccessor:
        def __init__(self, obj):
            self._obj = obj

        def __call__(self):
            return 'accessor'

    assert Target.__dict__['existing_method'] is not original_method
    pv.unregister_dataset_accessor('existing_method', Target)
    assert Target.__dict__['existing_method'] is original_method


def test_unregister_missing_raises():
    with pytest.raises(ValueError, match='No registered accessor'):
        pv.unregister_dataset_accessor('never_registered', pv.PolyData)


def test_unregister_inherited_accessor_raises():
    """An accessor inherited from a parent must be unregistered on the parent."""

    @pv.register_dataset_accessor('parent_only', pv.DataSet)
    class ParentAccessor:
        def __init__(self, dataset):
            self._dataset = dataset

    assert hasattr(pv.Sphere(), 'parent_only')
    with pytest.raises(ValueError, match='No registered accessor'):
        pv.unregister_dataset_accessor('parent_only', pv.PolyData)

    pv.unregister_dataset_accessor('parent_only', pv.DataSet)


def test_unregister_updates_records():
    @pv.register_dataset_accessor('tracked', pv.PolyData)
    class TrackedAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    names_before = [r.name for r in pv.registered_accessors()]
    assert 'tracked' in names_before

    pv.unregister_dataset_accessor('tracked', pv.PolyData)
    names_after = [r.name for r in pv.registered_accessors()]
    assert 'tracked' not in names_after


def test_registered_accessors_reports_records():
    @pv.register_dataset_accessor('introspect_me', pv.PolyData)
    class IntrospectMeAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    records = pv.registered_accessors()
    matches = [r for r in records if r.name == 'introspect_me']
    assert len(matches) == 1
    record = matches[0]
    assert record.target is pv.PolyData
    assert record.accessor is IntrospectMeAccessor
    assert record.source.endswith('IntrospectMeAccessor')


def test_registered_accessors_returns_tuple():
    assert isinstance(pv.registered_accessors(), tuple)


def test_re_register_replaces_single_record():
    @pv.register_dataset_accessor('single_record', pv.PolyData)
    class FirstAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        @pv.register_dataset_accessor('single_record', pv.PolyData)
        class SecondAccessor:
            def __init__(self, mesh):
                self._mesh = mesh

    records = [r for r in pv.registered_accessors() if r.name == 'single_record']
    assert len(records) == 1
    assert records[0].accessor is SecondAccessor


def test_save_restore_round_trip_empty():
    state = _reg_mod._save_registry_state()
    _reg_mod._restore_registry_state(state)
    assert pv.registered_accessors() == ()


def test_save_restore_round_trip_populated():
    @pv.register_dataset_accessor('snapshot_me', pv.PolyData)
    class SnapshotAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    state = _reg_mod._save_registry_state()

    @pv.register_dataset_accessor('post_snapshot', pv.PolyData)
    class PostSnapshotAccessor:
        def __init__(self, mesh):
            self._mesh = mesh

    pv.unregister_dataset_accessor('snapshot_me', pv.PolyData)

    _reg_mod._restore_registry_state(state)

    names = {r.name for r in pv.registered_accessors()}
    assert 'snapshot_me' in names
    assert 'post_snapshot' not in names
    assert hasattr(pv.Sphere(), 'snapshot_me')
    assert not hasattr(pv.Sphere(), 'post_snapshot')


def test_save_restore_round_trip_preserves_override():
    """Snapshot, override, then restore — the inherited attribute is visible again."""
    original_clip = pv.PolyData.clip
    state = _reg_mod._save_registry_state()

    @pv.register_dataset_accessor('clip', pv.PolyData, override=True)
    class ClipOverride:
        def __init__(self, mesh):
            self._mesh = mesh

    _reg_mod._restore_registry_state(state)
    assert pv.PolyData.clip is original_clip
    assert 'clip' not in pv.PolyData.__dict__


def test_fixture_isolation_setup():
    """Register an accessor; the conftest fixture must clean it up."""

    @pv.register_dataset_accessor('leaked_accessor', pv.PolyData)
    class LeakedAccessor:
        def __init__(self, mesh):
            self._mesh = mesh


def test_fixture_isolation_teardown():
    """The accessor from the previous test should not be visible."""
    assert not hasattr(pv.Sphere(), 'leaked_accessor')
    assert 'leaked_accessor' not in {r.name for r in pv.registered_accessors()}


def test_protocol_matches_valid_accessor():
    class Good:
        def __init__(self, dataset):
            self._dataset = dataset

    assert isinstance(Good(None), pv.DataSetAccessor)


def test_slotted_target_falls_back_to_uncached():
    """A target class without ``__dict__`` still gets a working accessor.

    Without per-instance caching, two accesses return distinct accessor
    instances.
    """

    class SlottedTarget:
        __slots__ = ()

    @pv.register_dataset_accessor('slotted_acc', SlottedTarget)
    class SlottedAccessor:
        def __init__(self, obj):
            self._obj = obj

        def value(self):
            return 7

    instance = SlottedTarget()
    assert instance.slotted_acc.value() == 7
    first = instance.slotted_acc
    second = instance.slotted_acc
    assert first is not second


def test_chaining_with_core_filters():
    @pv.register_dataset_accessor('double_points', pv.PolyData)
    class DoublePointsAccessor:
        """Toy accessor that returns a translated copy of the mesh."""

        def __init__(self, mesh):
            self._mesh = mesh

        def translated(self, dx=1.0):
            out = self._mesh.copy()
            out.translate((dx, 0.0, 0.0), inplace=True)
            return out

    original = pv.Sphere()
    result = original.clean().double_points.translated(dx=2.0).decimate(0.5)
    assert isinstance(result, pv.PolyData)
    assert result.n_points <= original.n_points
