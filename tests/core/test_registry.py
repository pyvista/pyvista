"""Tests for the machinery shared by every plugin registry."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import pytest

from pyvista.core.utilities._registry import EntryPointRefusedError
from pyvista.core.utilities._registry import _EntryPointLoader
from pyvista.core.utilities._registry import _HandlerTable
from pyvista.core.utilities._registry import handler_source


def _entry_point(name, value='pkg:obj', group='demo.group'):
    ep = MagicMock()
    ep.name = name
    ep.value = value
    ep.group = group
    return ep


def _loader(eps, *, load=None, key=None, groups=('demo.group',)):
    """Build a loader over ``eps`` whose scan sees nothing else."""
    loaded = []

    def _record(name, pending):
        loaded.append((name, pending[0].value))

    loader = _EntryPointLoader(
        *groups,
        key=key or (lambda ep, _group: ep.name),
        load=load or _record,
        keep_causes=True,
    )
    by_group = {}
    for ep in eps:
        by_group.setdefault(ep.group, []).append(ep)
    patcher = patch(
        'pyvista.core.utilities._registry.entry_points',
        side_effect=lambda *, group: by_group.get(group, []),
    )
    return loader, loaded, patcher


def test_scan_reads_metadata_without_loading():
    loader, loaded, patcher = _loader([_entry_point('a'), _entry_point('b')])
    with patcher as entry_points:
        loader.ensure_scanned()
        loader.ensure_scanned()
    entry_points.assert_called_once_with(group='demo.group')
    assert loader.loaded is True
    assert loader.pending_keys() == ('a', 'b')
    assert loaded == []


def test_scan_skips_keys_the_key_callback_rejects():
    loader, _, patcher = _loader(
        [_entry_point('keep'), _entry_point('drop')],
        key=lambda ep, _group: None if ep.name == 'drop' else ep.name,
    )
    with patcher:
        assert loader.pending_keys() == ('keep',)


def test_scan_covers_every_group_and_drops_a_double_declaration():
    eps = [
        _entry_point('a', group='first'),
        _entry_point('a', group='second'),
        _entry_point('b', value='pkg:b', group='second'),
    ]
    seen = []
    loader, _, patcher = _loader(
        eps,
        groups=('first', 'second'),
        key=lambda ep, group: seen.append((ep.name, group)) or ep.name,
    )
    with patcher:
        loader.ensure_scanned()
    assert seen == [('a', 'first'), ('b', 'second')]
    assert loader.pending['a'] == [eps[0]]


def test_resolve_loads_once_and_removes_the_key():
    loader, loaded, patcher = _loader([_entry_point('a')])
    with patcher:
        assert loader.resolve('a') is True
        assert loader.resolve('a') is False
        assert loader.resolve('missing') is False
    assert loaded == [('a', 'pkg:obj')]
    assert loader.pending == {}


def test_failed_load_warns_once_stays_pending_and_is_hidden():
    load = MagicMock(side_effect=ImportError('missing dep'))
    loader, _, patcher = _loader([_entry_point('a', value='pkg:broken')], load=load)
    with patcher:
        with pytest.warns(
            UserWarning, match='Failed to load demo.group entry point "a" from pkg:broken'
        ):
            assert loader.resolve('a') is False
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for _ in range(3):
                assert loader.resolve('a') is False
        assert loader.pending_keys() == ()
    assert load.call_count == 1
    assert 'a' in loader.pending
    assert 'missing dep' in loader.failure('a')
    assert isinstance(loader.take_cause('a'), ImportError)
    assert loader.take_cause('a') is None


def test_resolve_all_retries_a_failed_plugin():
    load = MagicMock(side_effect=[ImportError('missing dep'), None])
    loader, _, patcher = _loader([_entry_point('a')], load=load)
    with patcher:
        with pytest.warns(UserWarning, match='Failed to load'):
            loader.resolve('a')
        loader.resolve_all()
    assert load.call_count == 2
    assert loader.pending == {}
    assert loader.failed == {}


def test_ensure_loaded_does_not_retry_a_failed_plugin():
    load = MagicMock(side_effect=ImportError('missing dep'))
    loader, _, patcher = _loader([_entry_point('a'), _entry_point('b')], load=load)
    with patcher:
        with pytest.warns(UserWarning, match='Failed to load'):
            loader.ensure_loaded()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            loader.ensure_loaded()
    assert load.call_count == 2


def test_refused_entry_point_propagates_and_stays_pending():
    def _refuse(name, _eps):
        msg = f'{name} refused'
        raise EntryPointRefusedError(msg)

    loader, _, patcher = _loader([_entry_point('a')], load=_refuse)
    with patcher:
        with pytest.raises(EntryPointRefusedError, match='a refused'):
            loader.resolve('a')
        with pytest.warns(UserWarning, match='a refused'):
            loader.resolve_all()
    assert 'a' in loader.pending
    assert loader.failed == {}


def test_reentrant_resolve_from_inside_the_load_returns_false():
    inner = []

    def _load(name, _eps):
        inner.append(loader.resolve(name))

    loader, _, patcher = _loader([_entry_point('a')], load=_load)
    with patcher:
        assert loader.resolve('a') is True
    assert inner == [False]
    assert loader.pending == {}
    assert loader.failed == {}


def test_concurrent_first_resolve_loads_once_and_both_threads_see_it():
    load = MagicMock(side_effect=lambda *_: time.sleep(0.2))
    loader, _, patcher = _loader([_entry_point('a')], load=load)
    barrier = threading.Barrier(2)
    results = []

    def _resolve():
        barrier.wait()
        results.append(loader.resolve('a'))

    with patcher:
        threads = [threading.Thread(target=_resolve) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert load.call_count == 1
    assert sorted(results) == [False, True]
    assert loader.pending == {}


def test_concurrent_first_scan_runs_once():
    loader, _, patcher = _loader([_entry_point('a')])
    barrier = threading.Barrier(4)

    def _scan():
        barrier.wait()
        loader.ensure_scanned()

    with patcher as entry_points:
        threads = [threading.Thread(target=_scan) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert entry_points.call_count == 1


def test_resolve_warns_when_several_entry_points_claim_a_key():
    eps = [_entry_point('a', value='pkg:first'), _entry_point('a', value='pkg:second')]
    loader, loaded, patcher = _loader(eps)
    with patcher:
        with pytest.warns(
            UserWarning,
            match='Multiple demo.group entry points registered for "a": pkg:first, pkg:second. '
            'Using pkg:first.',
        ):
            assert loader.resolve('a') is True
    assert loaded == [('a', 'pkg:first')]


def test_snapshot_and_reset_round_trip():
    load = MagicMock(side_effect=ImportError('missing dep'))
    loader, _, patcher = _loader([_entry_point('a'), _entry_point('b')], load=load)
    with patcher:
        with pytest.warns(UserWarning, match='Failed to load'):
            loader.resolve('a')
        state = loader.snapshot()
        loader.reset()
        assert loader.loaded is False
        assert loader.pending == {}
        assert loader.failed == {}
        loader.restore(state)
    assert loader.loaded is True
    assert list(loader.pending) == ['a', 'b']
    assert list(loader.failed) == ['a']
    assert loader.take_cause('a') is None


def test_handler_table_records_sources_and_warns_on_replacement():
    table = _HandlerTable(
        'demo', builtin=lambda key: key == 'builtin', builtin_label='a stock demo'
    )
    assert not table
    table.set('a', 1, 'pkg:one')
    assert 'a' in table
    assert table.get('a') == 1
    assert table.source('a') == 'pkg:one'
    assert table.source('missing') == '<unknown>'
    with pytest.warns(
        UserWarning,
        match='Registering demo "a" replaces an existing custom registration from pkg:one',
    ):
        table.register('a', handler_source)
    assert table.source('a') == 'pyvista.core.utilities._registry.handler_source'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        table.register('a', 1, 'pkg:one', override=True)
        table.register('b', 2, 'pkg:two')
    with pytest.raises(ValueError, match=r'collides with a stock demo\. Use override=True'):
        table.register('builtin', 3)
    table.register('builtin', 3, override=True)
    assert table.pop('builtin') == 3
    assert table.items() == [('a', 1), ('b', 2)]
    assert table.keys() == {'a', 'b'}
    assert list(table) == ['a', 'b']
    assert len(table) == 2
    state = table.snapshot()
    assert table.pop('a') == 1
    assert table.pop('a') is None
    table.restore(state)
    assert table.get('a') == 1
    table.clear()
    assert not table


def test_handler_source():
    assert handler_source(handler_source) == 'pyvista.core.utilities._registry.handler_source'
    assert handler_source(object()).startswith('<unknown>.<object object')
