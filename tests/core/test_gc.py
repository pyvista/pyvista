"""The core leak check must fail on a leak only it can see.

``tests/plotting/test_gc.py`` already covers the leaks both checks catch. What
is left for core is the one its own policy exists for: it does not sweep VTK's
ghost map before reporting, and that sweep is what let #8873 through.

``expect_check_gc_fail`` asserts the teardown check *fails*, so a change that
quietly stops detecting the leak turns the test red.

The machinery both suites share is tested in ``tests/test_gc_check.py``.
"""

from __future__ import annotations

import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.core._vtk_utilities import _SETDATA_TAKES_OWNERSHIP

pytestmark = pytest.mark.check_gc


@pytest.mark.skipif(
    not _SETDATA_TAKES_OWNERSHIP,
    reason='CellArray holds its own arrays on VTK < 9.6, so the check never runs there',
)
@pytest.mark.expect_check_gc_fail
def test_leak_ghosted_attribute_dict() -> None:
    """A VTK object stashed on a wrapper whose C++ object outlives it.

    This is the shape of #8873. VTK keeps the attribute dict of a wrapper that
    dies while its C++ object is still referenced, so it can restore it if that
    object resurfaces in Python, and anything the dict holds stays alive with
    it. The map is only swept when a new ghost is added, so the entry outlives
    the mesh. Planted under the plotting policy, which sweeps first, this leak
    does not fail the check at all.
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
