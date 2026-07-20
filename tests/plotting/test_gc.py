from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

import pyvista as pv
from pyvista import _vtk


@pytest.mark.expect_check_gc_fail
def test_leak_vtk() -> None:
    """Create a vtk leak with a simple self-reference."""
    sphere = _vtk.vtkSphereSource()
    sphere.self_ref = sphere


@pytest.mark.expect_check_gc_fail
def test_leak_pv(sphere) -> None:
    """A VTK leak within a pyvista object with a simple self-reference."""
    points = sphere.points
    points.VTKObject._ref = points


def test_run_leak_tests(tmp_path: Path) -> None:
    shutil.copy(Path(__file__).parent / 'conftest.py', tmp_path / 'conftest.py')

    test_file = tmp_path / 'test_leak_pv.py'
    test_file.write_text("""
import pyvista as pv


def test_leak_pv() -> None:
    sphere = pv.Sphere()
    points = sphere.points
    points.VTKObject._ref = points
""")

    # The failure report contains non-ASCII box-drawing characters (refleak's
    # referrer tree), so force UTF-8 on both sides of the pipe -- otherwise
    # the runner's locale (ASCII on CI) breaks the encode or decode step.
    result = subprocess.run(
        ['pytest', '-v', str(test_file)],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        encoding='utf-8',
        errors='replace',
    )

    assert result.returncode != 0
    # Matches the singular and the plural: how many objects the planted cycle
    # keeps alive is a VTK implementation detail.
    assert 'new VTK/plotter object' in result.stdout


@pytest.mark.expect_check_gc_fail
def test_leak_pv_plotter() -> None:
    """Trigger a leak in pyvista.Plotter by disabling cleanup ."""

    def noop(*args: Any, **kwargs: Any):  # noqa: ARG001
        return None

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.enable_point_picking()
    pl.close = noop
    pl.deep_clean = noop
