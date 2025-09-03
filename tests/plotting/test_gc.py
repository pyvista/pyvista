from __future__ import annotations

from typing import Any

import pytest
import vtk

import pyvista as pv


@pytest.mark.expect_check_gc_fail
def test_leak_vtk() -> None:
    """Create a vtk leak with a simple self-reference."""
    sphere = vtk.vtkSphereSource()
    sphere.self_ref = sphere


@pytest.mark.parametrize(
    'decorator, expected_exit',
    [
        ('@pytest.mark.expect_check_gc_fail', pytest.ExitCode.OK),
        ('', pytest.ExitCode.TESTS_FAILED),
    ],
)
def test_expect_check_gc_fail(
    pytester: pytest.Pytester,
    decorator: str,
    expected_exit: ExitStatus,
):
    tests = f"""
    import pytest

    {decorator}
    def test_leak_pv(sphere) -> None:
        points = sphere.points
        points.VTKObject._ref = points.VTKObject
    """
    p = pytester.makepyfile(tests)
    results = pytester.runpytest(p)

    # Check pytest exit code
    assert results.ret == expected_exit

    if decorator:
        # With the marker, the test should pass
        results.assert_outcomes(passed=1)
    else:
        # Without the marker, the leak should trigger failure
        results.assert_outcomes(failed=1)


@pytest.mark.expect_check_gc_fail
def test_leak_pv_plotter() -> None:
    """Trigger a leak in pyvista.Plotter by disabling cleanup ."""

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.enable_point_picking()
    pl.close = noop
    pl.deep_clean = noop
