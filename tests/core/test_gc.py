"""The core leak check must fail on a leak only it can see.

``tests/plotting/test_gc.py`` already covers the leaks both checks catch. What
is left for core is the one its own policy exists for: it does not sweep VTK's
ghost map before reporting, and that sweep is what let #8873 through. Planted
under the plotting policy, the leak below does not fail the check at all.

``expect_check_gc_fail`` asserts the teardown check *fails*, so a change that
quietly stops detecting the leak turns the test red.
``test_no_leak_when_nothing_is_held`` is what stops that passing on a check
that simply fails on everything.

The machinery both suites share is tested in ``tests/test_gc_check.py``.
"""

from __future__ import annotations

import pytest

import pyvista as pv
from pyvista import _vtk

pytestmark = pytest.mark.check_gc


@pytest.mark.expect_check_gc_fail
def test_leak_ghosted_attribute_dict() -> None:
    """A VTK object stashed on a wrapper whose C++ object outlives it.

    This is the shape of #8873. VTK keeps the attribute dict of a wrapper that
    dies while its C++ object is still referenced, so it can restore it if that
    object resurfaces in Python, and anything the dict holds stays alive with
    it. The map is only swept when a new ghost is added, so the entry outlives
    the mesh, and a check that sweeps first sees nothing wrong.
    """
    polydata = _vtk.vtkPolyData()
    cells = _vtk.vtkCellArray()
    cells.stashed = _vtk.vtkPoints()
    polydata.SetPolys(cells)  # the C++ object now outlives the wrapper
    del cells


def test_no_leak_when_nothing_is_held() -> None:
    """The negative control: an ordinary mesh must not trip the check."""
    mesh = pv.Sphere()
    assert mesh.n_points
