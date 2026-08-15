"""Opt-in leak checking for core tests.

``tests/plotting/conftest.py`` runs a similar check on every plotting test, but sweeps
VTK's ghost map and re-checks before reporting -- plotter teardown leaves stale ghosts
behind, and 21 plotting tests rely on that forgiveness. The sweep also forgives a real
bug: pyvista stashing a VTK object on a wrapper that C++ owns, which then outlives the
mesh in the ghost ``__dict__``. That is how #8873 shipped a leak the MNE integration
tests caught but ours did not.

Modules owning dataset construction opt into the strict (no-sweep) check with::

    pytestmark = pytest.mark.check_gc

Opt-in rather than autouse because the check walks the heap twice per test -- 4x the
runtime over all of ``tests/core`` -- and ~70 core tests hold VTK objects past teardown
today. Widening the opt-in as those are cleaned up is the point.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from refleak.testing import Snapshot
from refleak.testing import gc_collect_once

from pyvista import _vtk
from pyvista.core._vtk_utilities import _SETDATA_TAKES_OWNERSHIP

_phase_report_key = pytest.StashKey()
_check_gc_key = pytest.StashKey()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    """Stash per-phase reports so the leak check can skip on failure."""
    outcome = yield
    rep = outcome.get_result()
    item.stash.setdefault(_phase_report_key, {})[rep.when] = rep


def _test_passed(item) -> bool:
    report = item.stash.get(_phase_report_key, {})
    return 'call' in report and report['call'].outcome == 'passed'


@pytest.fixture(autouse=True)
def check_gc(request):
    """Snapshot live VTK objects so leaks from this test can be detected."""
    node = request.node
    if (
        # On VTK < 9.6 CellArray must hold its own arrays, so this can never pass there
        not _SETDATA_TAKES_OWNERSHIP
        or node.get_closest_marker('check_gc') is None
        or node.get_closest_marker('skip_check_gc')
    ):
        yield
        return
    node.stash[_check_gc_key] = Snapshot(_vtk.vtkObjectBase, label='VTK')
    yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item):
    """Ensure every VTK object created during a test is collected by teardown."""
    yield
    snapshot = item.stash.get(_check_gc_key, None)
    if snapshot is None:
        return
    del item.stash[_check_gc_key]

    if not _test_passed(item):
        return

    # pytest holds fixture values in item.funcargs until after teardown hooks run, so a
    # VTK-typed fixture value would always look like a leak. The test passed and its
    # fixtures are finalized, so release them a moment early.
    item.funcargs.clear()

    request = SimpleNamespace(node=item)
    gc_collect_once(request)
    snapshot.assert_no_new(f'teardown of {item.name}', request=request)
