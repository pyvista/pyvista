"""Test the leak-check machinery in :mod:`tests.gc_check` itself.

The suites that use it can only exercise it through their own conftest, and each
of those supplies a different policy. What is checked here is common to both and
belongs to neither, so it drives ``take_snapshot`` and ``assert_no_leaks``
directly. Nothing under ``tests/`` at this level runs the check, so these are
outside the thing they are testing.

The leaks the check has to catch live with the policy that catches them:
``tests/plotting/test_gc.py`` and ``tests/core/test_gc.py``.
"""

from __future__ import annotations

import gc

from pyvista import _vtk
from tests import gc_check


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
    """
    # The stubs report no passing call phase, so the checks do their freeze
    # bookkeeping and skip the scan itself, which is what is under test here.
    outer, inner = _StubItem('outer'), _StubItem('inner')

    assert gc.get_freeze_count() == 0
    gc_check.take_snapshot(outer, _vtk.vtkObjectBase, 'VTK')
    frozen = gc.get_freeze_count()
    assert frozen > 0

    gc_check.take_snapshot(inner, _vtk.vtkObjectBase, 'VTK')
    gc_check.assert_no_leaks(inner, flush_ghosts=False)
    assert gc.get_freeze_count() == frozen, 'the inner check thawed the outer freeze'

    gc_check.assert_no_leaks(outer, flush_ghosts=False)
    assert gc.get_freeze_count() == 0


def test_leak_at_a_reused_address_is_still_found() -> None:
    """A leak is found even where it reuses the address of an object that died.

    The check this replaced recorded the ``id()`` of everything alive beforehand
    and reported what was not in that set, so a leak allocated at a dead
    object's address was indistinguishable from that dead object and passed
    silently. It is not a rare corner: CPython reuses the most recently freed
    block of a size class, so in 200 trials the address was reused 199 times and
    the leak was missed every one of them. Freezing has no ids to collide.
    """
    # Whether the allocator hands back the address it just freed is its own
    # business, so retry until it does rather than assert that it will. The leak
    # must be found on every pass; landing on the address is what makes a pass
    # worth having.
    for _ in range(20):
        doomed = _vtk.vtkPoints()
        address = id(doomed)

        gc.freeze()  # what the check does before a test runs, with doomed alive
        try:
            del doomed  # and it dies during the test, freeing its address
            leaked = _vtk.vtkPoints()
            reused = id(leaked) == address

            gc.collect()
            survivors = [obj for obj in gc.get_objects() if isinstance(obj, _vtk.vtkObjectBase)]
            assert any(obj is leaked for obj in survivors)
        finally:
            gc.unfreeze()

        del leaked
        if reused:
            break
