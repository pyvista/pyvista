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
    reason='the core check is disabled entirely on VTK < 9.6, see tests/core/conftest.py',
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

    The ghost goes on a command the mesh holds as an observer rather than on the
    mesh's own arrays because Kitware/VTK@641b2b68 (vtk/vtk!13226, VTK 9.7)
    evicts a ghost from a ``DeleteEvent`` observer, which only a ``vtkObject``
    can carry. A command is a ``vtkObjectBase`` that is not a ``vtkObject``, so
    its ghost still waits for the sweep.
    """
    polydata = _vtk.vtkPolyData()
    # The observer list holds the C++ command, so it outlives the wrapper below.
    tag = polydata.AddObserver('ModifiedEvent', lambda *_: None)
    command = polydata.GetCommand(tag)
    command.stashed = _vtk.vtkPoints()
    del command


def test_no_leak_when_nothing_is_held() -> None:
    """The negative control: an ordinary mesh must not trip the check."""
    mesh = pv.Sphere()
    assert mesh.n_points
