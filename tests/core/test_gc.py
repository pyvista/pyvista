"""The core leak check must fail on the leak shape that #8873 shipped.

``tests/plotting/test_gc.py`` covers the leaks both checks catch. What is left
for core is the one its policy exists for: a VTK object stashed on a wrapper
that C++ owns, which outlives the mesh in the ghost ``__dict__``.

``expect_check_gc_fail`` asserts the teardown check *fails*, so a change that
quietly stops detecting the leak turns the test red.

The machinery both suites share is tested in ``tests/test_gc_check.py``.
"""

from __future__ import annotations

import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.core._vtk_utilities import _SETDATA_TAKES_OWNERSHIP

# Built at import, so the leak check -- which freezes the heap at the start of each test
# -- sees it as pre-existing rather than as something the test below leaked. Its C++
# object is what keeps that test's ghost un-sweepable, and it has to outlive the test for
# that: this stands in for the plotter or the container that holds a mesh in real code.
_HOLDER = _vtk.vtkPolyData()
_HOLDER.SetPoints(_vtk.vtkPoints())


@pytest.mark.skipif(
    not _SETDATA_TAKES_OWNERSHIP,
    reason='the core check is disabled entirely on VTK < 9.6, see tests/core/conftest.py',
)
@pytest.mark.expect_check_gc_fail
def test_leak_ghosted_attribute_dict() -> None:
    """A VTK object stashed on a wrapper whose C++ object outlives it.

    This is the shape of #8873. VTK keeps the attribute dict of a wrapper that dies
    while its C++ object is still referenced -- a "ghost" -- so it can restore the
    attributes if that object resurfaces in Python, and anything the dict holds stays
    alive with it.

    The ghost is planted on a wrapper of ``_HOLDER``'s points rather than on a wrapper
    of something this test owns, because that is the difference between a leak and
    deferred bookkeeping: a ghost whose C++ object has died is swept out of the map by
    the next sweep, and the check forgives it (see :func:`tests.gc_check._flush_vtk_ghosts`).
    A ghost whose C++ object is still alive is swept by nothing and holds its contents
    for as long as that object lives, which is how #8873 leaked a mesh per plotter into
    a long-running session.
    """
    # A fresh wrapper of an object C++ owns; the attribute makes its dict worth ghosting
    points = _HOLDER.GetPoints()
    points.stashed = _vtk.vtkPoints()
    del points  # wrapper dies, its C++ object does not -- so the ghost is here to stay


def test_no_leak_when_nothing_is_held() -> None:
    """The negative control: an ordinary mesh must not trip the check."""
    mesh = pv.Sphere()
    assert mesh.n_points
