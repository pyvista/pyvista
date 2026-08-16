"""Shared leak-check machinery for the ``core`` and ``plotting`` conftests.

Both run the same check -- put the objects alive before a test out of reach, assert none
of the ones it creates outlive it -- so the mechanics live here and each conftest only
supplies its policy. They differ in exactly two ways:

* what they match: plotting also watches ``BasePlotter``, which is not a VTK object.
* ``flush_ghosts``: plotting forgives a leak that disappears once VTK's ghost map is
  swept, because plotter teardown routinely leaves stale ghosts behind. Core does not,
  and that strictness is the point -- the retry is what let the #8873 leak through.

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
#
# A list rather than a counter so the depth is mutated in place, which a module-level int
# would need ``global`` for. It holds the name of each nested check, so an unbalanced
# freeze names the tests involved rather than just failing to add up.
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
    the empty ``objs``. Recording ids instead meant a ``gc.collect()`` and a scan of the
    whole heap here and again at teardown, four passes over ~180k objects that between
    them cost more than the tests: ``tests/core`` runs the check over every one of its
    7,559 tests in less time than it took to run it over 397 of them this way.

    It is also stricter. An id-based snapshot cannot distinguish a new object allocated at
    a dead one's address from that dead one, and silently passes; there are no ids to
    reuse here.

    Matching by ``isinstance`` rather than class-name prefix also covers pyvista's own
    vtk subclasses (``PolyData``, ...) and the pythonic override subclasses VTK >= 9.6
    instantiates, whose names lack the ``vtk`` prefix. Passing several types as one tuple
    keeps this to a single pass.
    """
    if not _frozen_for:
        gc.freeze()
    _frozen_for.append(item.name)
    item.stash[_check_gc_key] = Snapshot(match, label=label, objs=[])


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

    # Whatever happens below, hand the frozen objects back to the collector. Leaving them
    # in the permanent generation would exempt them from collection for the rest of the
    # session, and the next test would see them as its own.
    try:
        _assert_no_leaks(item, snapshot, flush_ghosts=flush_ghosts, before_check=before_check)
    finally:
        # Never raise from here: this runs while a leak assertion may be in flight, and
        # an IndexError would replace it with something that explains nothing.
        if _frozen_for:
            _frozen_for.pop()
        if not _frozen_for:
            gc.unfreeze()


def _assert_no_leaks(
    item,
    snapshot: Snapshot,
    *,
    flush_ghosts: bool,
    before_check: Callable[[], None] | None,
) -> None:
    """Do the checking, with :func:`assert_no_leaks` owning the unfreeze around it."""
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

    def _assert_no_new():
        try:
            snapshot.assert_no_new(when, request=request)
        except AssertionError:
            if not flush_ghosts:
                raise
            # A stale VTK ghost is deferred bookkeeping, not a leak: flush the ghost map
            # and re-check before reporting a failure.
            _flush_vtk_ghosts()
            gc.collect()
            snapshot.assert_no_new(when, request=request)

    if item.get_closest_marker('expect_check_gc_fail'):
        with pytest.raises(AssertionError, match='Found '):
            _assert_no_new()
        return

    _assert_no_new()
