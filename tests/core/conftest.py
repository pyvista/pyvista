"""Opt-in leak checking for core tests.

Same machinery as the plotting conftest (see :mod:`tests.gc_check`), with one
deliberate difference: no ghost-map sweep before reporting. Plotting needs that
forgiveness for its own teardown, but it also forgives pyvista stashing a VTK object on
a wrapper that C++ owns, which then outlives the mesh in the ghost ``__dict__`` -- how
#8873 shipped a leak the MNE integration tests caught but ours did not.

Modules owning dataset construction opt in with::

    pytestmark = pytest.mark.check_gc

Opt-in rather than autouse only because ~70 core tests hold VTK objects past teardown
today; widening it as those are cleaned up is the point. Cost is no longer a reason:
since the check freezes the heap rather than scanning it, running it over all 7,559
tests here takes less time than running it over 397 of them did.
"""

from __future__ import annotations

import pytest

from pyvista import _vtk
from pyvista.core._vtk_utilities import _SETDATA_TAKES_OWNERSHIP
from tests.gc_check import assert_no_leaks
from tests.gc_check import stash_phase_report
from tests.gc_check import take_snapshot


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    """Stash per-phase reports so the leak check can skip on failure."""
    outcome = yield
    stash_phase_report(item, outcome.get_result())


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
    take_snapshot(node, _vtk.vtkObjectBase, 'VTK')
    yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item):
    """Ensure every VTK object created during a test is collected by teardown."""
    yield
    assert_no_leaks(item, flush_ghosts=False)
