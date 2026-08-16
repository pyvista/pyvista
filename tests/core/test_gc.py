"""The core leak check must fail on a real leak.

``tests/plotting/test_gc.py`` does this for the plotting check. Core runs the
stricter half of the same machinery -- no ghost sweep before reporting, because
the sweep is what let #8873 through -- and nothing held it to catching anything
until now.

Each test here plants a leak and is marked ``expect_check_gc_fail``, so the
teardown check is asserted to *fail*. A change that quietly stops detecting one
of these turns its test red.
"""

from __future__ import annotations

import gc

import pytest

import pyvista as pv
from pyvista import _vtk

pytestmark = pytest.mark.check_gc


@pytest.mark.expect_check_gc_fail
def test_leak_self_referencing_vtk_object() -> None:
    """A VTK object in a cycle with itself."""
    source = _vtk.vtkSphereSource()
    source.self_ref = source


@pytest.mark.expect_check_gc_fail
def test_leak_pyvista_wrapper() -> None:
    """The same, reached through a pyvista wrapper rather than a raw VTK one."""
    mesh = pv.Sphere()
    points = mesh.points
    points.VTKObject._ref = points


@pytest.mark.expect_check_gc_fail
def test_leak_ghosted_attribute_dict() -> None:
    """A VTK object stashed on a wrapper whose C++ object outlives it.

    This is the shape of #8873. VTK keeps the attribute dict of a wrapper that
    dies while its C++ object is still referenced, so it can restore it if that
    object resurfaces in Python, and anything the dict holds stays alive with
    it. The map is only swept when a new ghost is added, so the entry outlives
    the mesh -- which is why the core check does not sweep before reporting.
    """
    polydata = _vtk.vtkPolyData()
    cells = _vtk.vtkCellArray()
    cells.stashed = _vtk.vtkPoints()
    polydata.SetPolys(cells)  # the C++ object now outlives the wrapper
    del cells


@pytest.mark.skip_check_gc  # this drives the machinery itself, so it must not be inside it
def test_leak_at_a_reused_address_is_still_found() -> None:
    """A leak is found even where it reuses the address of an object that died.

    The check this replaced recorded the ``id()`` of everything alive beforehand
    and reported what was not in that set, so a leak allocated at a dead
    object's address was indistinguishable from that dead object and passed
    silently. It is not a rare corner: CPython reuses the most recently freed
    block of a size class, so in 200 trials the address was reused 199 times and
    the leak was missed every one of them. Freezing has no ids to collide.
    """
    # Whether the allocator hands back the address it just freed is its own
    # business, so retry until it does rather than assert that it will. The leak
    # must be found on every pass; landing on the address is what makes a pass
    # worth having.
    for _ in range(20):
        doomed = _vtk.vtkPoints()
        address = id(doomed)

        gc.freeze()  # what the check does before a test runs, with doomed alive
        try:
            del doomed  # and it dies during the test, freeing its address
            leaked = _vtk.vtkPoints()
            reused = id(leaked) == address

            gc.collect()
            survivors = [obj for obj in gc.get_objects() if isinstance(obj, _vtk.vtkObjectBase)]
            assert any(obj is leaked for obj in survivors)
        finally:
            gc.unfreeze()

        del leaked
        if reused:
            break


def test_no_leak_when_nothing_is_held() -> None:
    """The negative control: an ordinary mesh must not trip the check.

    Without this, every test above would still pass if the check simply failed
    on everything.
    """
    mesh = pv.Sphere()
    assert mesh.n_points
