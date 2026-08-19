"""Shared leak-check machinery for the repository-wide and ``plotting`` conftests.

Every test is checked -- put the objects alive before it out of reach, assert none of
the ones it creates outlive it. ``tests/conftest.py`` runs it for the repository;
``tests/plotting/conftest.py`` overrides the fixture for its own tests, because
plotting has to watch one more type (``BasePlotter``, which is not a VTK object) and
has to close its plotters before the check runs. The mechanics live here so the two
supply only that.

The check runs from a ``pytest_runtest_teardown`` hookwrapper rather than a fixture
finalizer: several fixtures are set up before an autouse fixture (``monkeypatch`` via
other autouse fixtures, the registry save/restore in ``tests/conftest.py``), so their
finalizers run *after* it -- and anything they still held would be misreported here.

.. warning::

    ``refleak``'s freeze mode keeps the heap frozen for the whole body of a covered test,
    so anything running in that body sees a *lying* collector: ``gc.get_referrers()``
    reports no referrers for an object that pre-dates the test, and ``gc.get_objects()``
    omits it entirely. This is not confined to pyvista's own code -- Hypothesis'
    ``register_random`` uses ``gc.get_referrers()`` to check that a PRNG handed to it is
    reachable, and wrongly concluded it was not (hence the warmup in
    ``tests/conftest.py``). Mark a test that needs a truthful view of the collector with
    ``@pytest.mark.skip_check_gc``, which costs it leak coverage.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
from refleak.testing import Snapshot
from refleak.testing import gc_collect_once

from pyvista import _vtk

_phase_report_key = pytest.StashKey()
_check_gc_key = pytest.StashKey()


def stash_phase_report(item, report) -> None:
    """Record a phase report so the check can be skipped when the test failed."""
    item.stash.setdefault(_phase_report_key, {})[report.when] = report


def check_enabled(node) -> bool:
    """Return whether this test should be checked for leaks.

    The check runs unless the test opts out with ``skip_check_gc`` or the whole run does
    with ``--no_check_gc``, which exists for local iteration only -- CI never passes it,
    and since the check freezes the heap rather than scanning it there is nothing to
    gain by it. ``expect_check_gc_fail`` overrides the flag: a test asserting that the
    check *fails* would otherwise pass while nothing ran.
    """
    if node.get_closest_marker('skip_check_gc'):
        return False
    if node.get_closest_marker('expect_check_gc_fail'):
        return True
    # A default rather than a hard lookup: a nested in-process session (``pytester``,
    # see tests/plotting/test_conftest.py) copies this conftest but not
    # ``tests/conftest.py``, which is what registers the option.
    return not node.config.getoption('no_check_gc', default=False)


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


def take_snapshot(item, match, label: str, owner: str) -> None:
    """Put every object alive now out of reach, so only *new* survivors are reported.

    ``owner`` records which conftest took the snapshot; see :func:`assert_no_leaks`.

    ``freeze=True`` has ``refleak`` call ``gc.freeze()``, moving them into the permanent
    generation, which the collector never walks and ``gc.get_objects()`` never reports --
    for the test body too, not just for the check, which is the hazard the module
    docstring warns about. So whatever the check finds afterwards was created by this
    test, and there is no "before" set to record: no collect and no heap scan happen here.
    Recording ids instead meant a collect and a scan of the whole heap here and again at
    teardown, four passes over ~180k objects that between them cost more than the tests.

    It is also stricter. An id-based snapshot cannot distinguish a new object allocated at
    a dead one's address from that dead one, and silently passes; there are no ids to
    reuse here.

    Matching by ``isinstance`` rather than class-name prefix also covers pyvista's own
    vtk subclasses (``PolyData``, ...) and the pythonic override subclasses VTK >= 9.6
    instantiates, whose names lack the ``vtk`` prefix. Passing several types as one tuple
    keeps this to a single pass.
    """
    item.stash[_check_gc_key] = (owner, Snapshot(match, label=label, freeze=True))


def assert_no_leaks(item, *, owner: str, flush_ghosts: bool) -> None:
    """Assert nothing matched by :func:`take_snapshot` outlived the test.

    ``owner`` is the conftest doing the checking. Two conftests register this hook --
    the repository-wide one and the plotting one, whose fixture overrides it for its own
    tests -- and both hooks run for a plotting test. Only the one whose fixture took the
    snapshot may check it: the other would report before ``tests/plotting`` has closed
    its plotters.
    """
    stashed = item.stash.get(_check_gc_key, None)
    if stashed is None or stashed[0] != owner:
        return
    snapshot = stashed[1]
    del item.stash[_check_gc_key]
    try:
        _assert_no_leaks(item, snapshot, flush_ghosts=flush_ghosts)
    finally:
        # Whatever happened above, hand the frozen objects back to the collector: leaving
        # them in the permanent generation would exempt them from collection for the rest
        # of the session, and the next test would see them as its own. A no-op on the
        # paths that reached the check, which thaws itself.
        snapshot.thaw()


def _assert_no_leaks(item, snapshot: Snapshot, *, flush_ghosts: bool) -> None:
    """Do the checking, with the heap frozen and the caller holding the thaw."""
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
        # ``refleak`` scans rather than re-checking after a failure, because the survivor
        # list it takes holds strong references and nothing can be freed once it is taken.
        _flush_vtk_ghosts()
        gc.collect()

    # assert_no_new takes its survivors while the heap is still frozen -- those are
    # exactly the objects this test created -- and thaws before reporting, on the failing
    # path too.
    if item.get_closest_marker('expect_check_gc_fail'):
        with pytest.raises(AssertionError, match='Found '):
            snapshot.assert_no_new(when, request=request)
        return

    snapshot.assert_no_new(when, request=request)
