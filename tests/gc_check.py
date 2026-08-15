"""Shared leak-check machinery for the ``core`` and ``plotting`` conftests.

Both run the same check -- snapshot the live VTK objects before a test, assert none of
them outlived it -- so the mechanics live here and each conftest only supplies its
policy. They differ in exactly two ways:

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
    """Record the live matching objects, so only *new* survivors are reported.

    Matching by ``isinstance`` rather than class-name prefix also covers pyvista's own
    vtk subclasses (``PolyData``, ...) and the pythonic override subclasses VTK >= 9.6
    instantiates, whose names lack the ``vtk`` prefix. Passing several types as one tuple
    keeps this to a single pass over the heap. ``Snapshot`` collects before it records
    ids, so no ``gc.collect()`` is needed here.
    """
    item.stash[_check_gc_key] = Snapshot(match, label=label)


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
