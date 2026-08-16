"""Shared leak-check machinery for the ``core`` and ``plotting`` conftests.

Both run the same check -- put the objects alive before a test out of reach, assert none
of the ones it creates outlive it -- so the mechanics live here and each conftest only
supplies its policy. They differ in exactly two ways:

* what they match: plotting also watches ``BasePlotter``, which is not a VTK object.
* ``flush_ghosts``: plotting sweeps VTK's ghost map before scanning, forgiving a leak
  that the sweep clears, because plotter teardown routinely leaves stale ghosts behind.
  Core does not, and that strictness is the point -- the sweep is what hid #8873.

The check runs from a ``pytest_runtest_teardown`` hookwrapper rather than a fixture
finalizer: several fixtures are set up before an autouse fixture (``monkeypatch`` via
other autouse fixtures, the registry save/restore in ``tests/conftest.py``), so their
finalizers run *after* it -- and anything they still held would be misreported here.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from refleak.testing import Snapshot
from refleak.testing import gc_collect_once

from pyvista import _vtk

if TYPE_CHECKING:
    from collections.abc import Callable

_phase_report_key = pytest.StashKey()
_check_gc_key = pytest.StashKey()

# Freezing is process-wide, and ``pytester`` runs a nested session in this same process
# with a copy of the conftest (see tests/plotting/test_conftest.py), so an inner test
# would otherwise unfreeze the heap out from under the outer one -- which then sees every
# object in the process as its own. Only the outermost check freezes and thaws.
_frozen_for: list[str] = []


def stash_phase_report(item, report) -> None:
    """Record a phase report so the check can be skipped when the test failed."""
    item.stash.setdefault(_phase_report_key, {})[report.when] = report


def _test_passed(item) -> bool:
    report = item.stash.get(_phase_report_key, {})
    return 'call' in report and report['call'].outcome == 'passed'


def _flush_vtk_ghosts() -> None:
    """Sweep dead entries out of VTK's ghost map.

    When a wrapper with attributes dies while its C++ object is still referenced, VTK
    "ghosts" the attribute dict so it can be restored should the C++ object resurface in
    Python. The map is only swept when a new ghost is added, so the dict of a wrapper
    that died during this test can linger after its C++ object dies -- and anything it
    holds (e.g. a composite mapper's ``_dataset``) then looks like a leak. Adding one
    throwaway ghost forces the sweep.
    """
    holder = _vtk.vtkPolyData()
    bait = _vtk.vtkPoints()
    bait._pyvista_ghost_bait = True
    holder.SetPoints(bait)
    # bait's wrapper dies while its C++ object is still held by holder, so it is added to
    # the ghost map, sweeping out stale ghosts; deleting holder then kills the C++ object,
    # letting a later sweep remove the bait itself.
    del bait
    del holder


def take_snapshot(item, match, label: str) -> None:
    """Put every object alive now out of reach, so only *new* survivors are reported.

    ``gc.freeze()`` moves them into the permanent generation, which the collector never
    walks and ``gc.get_objects()`` never reports. So whatever the check finds afterwards
    was created by this test, and the snapshot needs no "before" set to subtract -- hence
    the empty ``objs``, which also means ``Snapshot``'s own ``collect=True`` is ignored
    and the ``gc.collect()`` it used to run here no longer happens. Recording ids instead
    meant that collect and a scan of the whole heap here and again at teardown, four
    passes over ~180k objects that between them cost more than the tests: ``tests/core``
    now runs the check over every one of its tests in less time than it took to run it
    over the 397 it covered this way.

    It is also stricter. An id-based snapshot cannot distinguish a new object allocated at
    a dead one's address from that dead one, and silently passes; there are no ids to
    reuse here.

    Matching by ``isinstance`` rather than class-name prefix also covers pyvista's own
    vtk subclasses (``PolyData``, ...) and the pythonic override subclasses VTK >= 9.6
    instantiates, whose names lack the ``vtk`` prefix. Passing several types as one tuple
    keeps this to a single pass.
    """
    # Built before the freeze: a raise from Snapshot would otherwise leave a name in
    # _frozen_for that nothing pops, and the worker's heap frozen for the rest of the run.
    snapshot = Snapshot(match, label=label, objs=[])
    if not _frozen_for:
        gc.freeze()
    _frozen_for.append(item.name)
    item.stash[_check_gc_key] = snapshot


def assert_no_leaks(
    item,
    *,
    flush_ghosts: bool,
    before_check: Callable[[], None] | None = None,
) -> None:
    """Assert nothing matched by :func:`take_snapshot` outlived the test."""
    snapshot = item.stash.get(_check_gc_key, None)
    if snapshot is None:
        return
    del item.stash[_check_gc_key]

    thawed = False

    def thaw() -> None:
        """Hand the frozen objects back to the collector, at most once.

        Leaving them in the permanent generation would exempt them from collection for
        the rest of the session, and the next test would see them as its own. The check
        thaws partway through (see :func:`_assert_no_leaks`); this is also the backstop
        for the paths that return or raise before getting that far.
        """
        # Never raise from here: this runs while a leak assertion may be in flight, and
        # an IndexError would replace it with something that explains nothing.
        nonlocal thawed
        if thawed:
            return
        thawed = True
        if _frozen_for:
            _frozen_for.pop()
        if not _frozen_for:
            gc.unfreeze()

    try:
        _assert_no_leaks(
            item, snapshot, flush_ghosts=flush_ghosts, before_check=before_check, thaw=thaw
        )
    finally:
        thaw()


def _assert_no_leaks(
    item,
    snapshot: Snapshot,
    *,
    flush_ghosts: bool,
    before_check: Callable[[], None] | None,
    thaw: Callable[[], None],
) -> None:
    """Do the checking, calling ``thaw`` once the survivors have been taken."""
    if before_check is not None:
        before_check()

    if not _test_passed(item):
        return

    # pytest holds every fixture value in item.funcargs until after all teardown hooks
    # have run (its runner only then sets it to None), so a VTK-typed fixture value
    # (sphere, texture, ...) would always be flagged as a leak. The test passed and its
    # fixtures are already finalized, so release them a moment early.
    item.funcargs.clear()

    when = f'teardown of {item.name}'
    # gc_collect_once deduplicates on request.node; it only needs .node
    request = SimpleNamespace(node=item)
    gc_collect_once(request)

    if flush_ghosts:
        # A stale VTK ghost is deferred bookkeeping, not a leak: sweep the map before
        # scanning rather than re-checking after a failure, because the survivor list
        # below holds strong references and nothing can be freed once it is taken.
        _flush_vtk_ghosts()
        gc.collect()

    # Take the survivors while the heap is still frozen -- these are exactly the objects
    # this test created. Thaw before reporting: gc.get_referrers() does not look in the
    # permanent generation either, so a leak anchored in a container that pre-dates the
    # test would have no visible referrer, and refleak drops a survivor it cannot explain.
    objs = gc.get_objects()
    thaw()

    if item.get_closest_marker('expect_check_gc_fail'):
        with pytest.raises(AssertionError, match='Found '):
            snapshot.assert_no_new(when, request=request, objs=objs)
        return

    snapshot.assert_no_new(when, request=request, objs=objs)
