from __future__ import annotations

from typing import Any

import pytest
import vtk

import pyvista as pv


def noop(*args: Any, **kwargs: Any):  # noqa: ARG001
    return None


@pytest.mark.expect_check_gc_fail
def test_leak_vtk() -> None:
    """Create a vtk leak with a simple self-reference."""
    sphere = vtk.vtkSphereSource()
    sphere.self_ref = sphere


@pytest.mark.expect_check_gc_fail
def test_leak_pv(sphere) -> None:
    """A VTK leak within a pyvista object with a simple self-reference."""
    points = sphere.points
    points.VTKObject._ref = points.VTKObject


@pytest.mark.expect_check_gc_fail
def test_leak_pv_plotter() -> None:
    """Trigger a leak in pyvista.Plotter by disabling cleanup ."""

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.enable_point_picking()
    pl.close = noop
    pl.deep_clean = noop
