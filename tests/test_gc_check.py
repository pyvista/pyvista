"""Test the leak-check machinery in :mod:`tests.gc_check` itself.

The conftests can only exercise it through a real test run, so what is checked
here drives ``take_snapshot`` and ``assert_no_leaks`` directly, on stub items.
The module opts out of the ambient check for that reason (see the marker below):
these tests have to be outside the thing they are testing.

The leaks the check has to catch live with the policy that catches them:
``tests/plotting/test_gc.py`` and ``tests/core/test_gc.py``.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
from refleak.testing import _core as _refleak_core

from pyvista import _vtk
from tests import gc_check

# These drive the freeze machinery by hand, on stub items, and the fixture below resets
# the process-wide freeze state afterwards. The ambient check from ``tests/conftest.py``
# holds a freeze across each of them, so it has to stay out of the way here -- these
# tests have to be outside the thing they are testing.
pytestmark = pytest.mark.skip_check_gc

_OWNER = 'tests.test_gc_check'


@pytest.fixture(autouse=True)
def _restore_freeze_state():
    """Leave the collector as we found it, however the test ended.

    These tests drive ``take_snapshot`` and ``assert_no_leaks`` by hand, and the
    freeze bookkeeping between them is process-wide state inside ``refleak``. A
    test that fails partway leaves it holding a freeze that nothing releases, so
    the heap stays frozen and every later check in this worker stops freezing and
    thawing -- degrading quietly rather than failing. The real conftest cannot
    strand it that way, because it takes the snapshot in setup and asserts in
    teardown, which pytest always runs.
    """
    yield
    # refleak counts nested freezes in a module global, and only the outermost thaw
    # unfreezes; a stranded count would keep every later thaw in this worker from
    # reaching gc.unfreeze(). Nothing public resets it, hence the reach inside.
    _refleak_core._freeze_depth = 0
    gc.unfreeze()


class _StubItem:
    """The parts of a pytest item the freeze bookkeeping touches."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.stash: dict = {}
        self.funcargs: dict = {}

    def get_closest_marker(self, _name: str) -> None:
        return None


def test_nested_checks_thaw_only_once() -> None:
    """An inner check must not thaw the heap the outer one froze.

    ``pytester`` runs a nested session in this same process with a copy of the
    conftest, so the checks nest for real (see tests/plotting/test_conftest.py).
    Freezing is process-wide, so an inner thaw hands the outer test the whole
    process to account for, and it reports every VTK object in it as a leak.
    ``refleak`` counts the nesting; this is the check that pyvista still gets the
    behavior it depends on.
    """
    # The stubs report no passing call phase, so the checks do their freeze
    # bookkeeping and skip the scan itself, which is what is under test here.
    outer, inner = _StubItem('outer'), _StubItem('inner')

    # A delta, not an absolute: CPython freezes a few hundred objects of its own
    # during startup, and other tests in this worker may have frozen more.
    baseline = gc.get_freeze_count()
    gc_check.take_snapshot(outer, _vtk.vtkObjectBase, 'VTK', owner=_OWNER)
    frozen = gc.get_freeze_count()
    assert frozen > baseline

    gc_check.take_snapshot(inner, _vtk.vtkObjectBase, 'VTK', owner=_OWNER)
    gc_check.assert_no_leaks(inner, owner=_OWNER, flush_ghosts=False)
    # Still frozen, rather than still exactly ``frozen``. The count drops by one
    # for every object that was alive at the freeze and dies while it is held,
    # which is ordinary refcounting; on a 1.1M object heap a drift of one is
    # routine and is not a thaw. A thaw empties the permanent generation
    # outright, so it shows up as 0, which this still catches.
    assert gc.get_freeze_count() > baseline, 'the inner check thawed the outer freeze'

    gc_check.assert_no_leaks(outer, owner=_OWNER, flush_ghosts=False)
    # Zero, not back to ``baseline``: thawing is not the inverse of freezing.
    # ``gc.unfreeze()`` empties the permanent generation outright, so it also
    # releases whatever was in it before the outer check froze anything.
    assert gc.get_freeze_count() == 0


def test_leak_at_a_reused_address_is_still_found() -> None:
    """A leak is found even where it reuses the address of an object that died.

    The check this replaced recorded the ``id()`` of everything alive beforehand
    and reported what was not in that set, so a leak allocated at a dead
    object's address was indistinguishable from that dead object and passed
    silently. It is not a rare corner: CPython reuses the most recently freed
    block of a size class, so in 200 trials the address was reused 199 times and
    the leak was missed every one of them. Freezing has no ids to collide.

    This is also the only test here that reaches the reporting code with an
    assertion in flight, so it is what covers the thaw on that path.
    """
    item = _StubItem('reused')
    gc_check.stash_phase_report(item, SimpleNamespace(when='call', outcome='passed'))

    doomed = _vtk.vtkPoints()
    address = id(doomed)

    # Nothing may raise between here and the check: it holds the freeze, and an
    # escape would leave the worker's heap frozen for every test after this one.
    gc_check.take_snapshot(item, _vtk.vtkObjectBase, 'VTK', owner=_OWNER)
    del doomed  # dies during the "test", freeing its address for the leak below
    leaked = _vtk.vtkPoints()
    leaked.self_ref = leaked  # a referrer for the report to walk

    with pytest.raises(AssertionError, match='Found 1 new VTK object'):
        gc_check.assert_no_leaks(item, owner=_OWNER, flush_ghosts=False)

    assert gc.get_freeze_count() == 0, 'the failing check left the heap frozen'
    assert id(leaked) == address, 'the allocator did not hand the address back'
    leaked.self_ref = None  # break the cycle, so the leak does not outlive its test
    del leaked


@pytest.mark.parametrize(
    ('markers', 'opted_out', 'expected'),
    [
        ((), False, True),
        ((), True, False),
        (('skip_check_gc',), False, False),
        # A test asserting that the check fails runs it even when the run opted out.
        (('expect_check_gc_fail',), True, True),
        # skip wins: a test that cannot survive a frozen heap cannot survive one
        # because it also expects the check to fail.
        (('skip_check_gc', 'expect_check_gc_fail'), False, False),
    ],
)
def test_check_enabled(markers, opted_out, expected) -> None:
    """Everything is checked but what ``--no_check_gc`` or a marker takes out."""
    node = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda _name, default=None: opted_out),  # noqa: ARG005
        get_closest_marker=lambda name: name if name in markers else None,
    )
    assert gc_check.check_enabled(node) is expected
