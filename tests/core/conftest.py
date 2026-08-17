"""Leak checking for every core test.

Same machinery, and now the same policy, as the plotting conftest (see
:mod:`tests.gc_check`): both sweep VTK's ghost map before scanning, and the sweep costs
no strictness. It removes an entry only when the C++ object behind it is already dead,
which makes the entry deferred bookkeeping -- the next sweep would drop it anyway. A
ghost whose C++ object is still alive, which is the shape #8873 shipped, survives the
sweep and is still reported (``tests/core/test_gc.py`` plants one).

Not sweeping cost 29 tests here, all of them pyvista wrapping its own state on a wrapper
that C++ owns (``MultiBlock._refs``, ``DataSetAttributes``, ``pyvista_ndarray``) and then
letting the wrapper die -- deferred every time, never an accumulation.
"""

from __future__ import annotations

import pytest

from pyvista import _vtk
from pyvista.core._vtk_utilities import _SETDATA_TAKES_OWNERSHIP
from tests.gc_check import assert_no_leaks
from tests.gc_check import check_enabled
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
        not _SETDATA_TAKES_OWNERSHIP or not check_enabled(node)
    ):
        yield
        return
    take_snapshot(node, _vtk.vtkObjectBase, 'VTK')
    yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item):
    """Ensure every VTK object created during a test is collected by teardown."""
    yield
    assert_no_leaks(item, flush_ghosts=True)
