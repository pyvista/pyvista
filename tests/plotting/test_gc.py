from __future__ import annotations

import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import weakref

import pytest

import pyvista as pv
from pyvista import _vtk

# vtk >= 9.8 sentinel: those wrappers traverse their instance dict (vtk/vtk!13603)
_VTK_GC_TRAVERSES_DICT = (9, 7, 99)

# Stands in for a module-level registry or cache: older than any test here, so the
# freeze puts it out of reach along with everything else alive at collection time.
_REGISTRY: list[_vtk.vtkPolyData] = []


@pytest.mark.expect_check_gc_fail
def test_leak_into_a_container_older_than_the_test() -> None:
    """A leak whose only referrer pre-dates the test must still be reported.

    ``gc.get_referrers`` skips the permanent generation just as ``gc.get_objects``
    does, so while the heap is frozen this object looks like it is held by nothing
    -- and the report drops a survivor it cannot explain. Every registry and cache
    that outlives a test has this shape, ``_ALL_PLOTTERS`` among them.
    """
    _REGISTRY.clear()  # whatever an earlier run of this test left, so it cannot pile up
    _REGISTRY.append(_vtk.vtkPolyData())


@pytest.mark.needs_vtk_version(
    less_than=_VTK_GC_TRAVERSES_DICT,
    reason='This VTK collects the cycle; see test_vtk_self_reference_collected.',
)
@pytest.mark.expect_check_gc_fail
def test_leak_vtk() -> None:
    """Create a vtk leak with a simple self-reference."""
    sphere = _vtk.vtkSphereSource()
    sphere.self_ref = sphere


@pytest.mark.needs_vtk_version(
    at_least=_VTK_GC_TRAVERSES_DICT,
    reason='This VTK cannot collect the cycle; see test_leak_vtk.',
)
def test_vtk_self_reference_collected() -> None:
    """A self-reference through a wrapper attribute is garbage-collectable."""
    sphere = _vtk.vtkSphereSource()
    sphere.self_ref = sphere
    ref = weakref.ref(sphere)
    del sphere
    gc.collect()
    assert ref() is None


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
    # PYTHONPATH: the copied conftest imports tests.gc_check, which is not
    # importable from tmp_path.
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-v', str(test_file)],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONPATH': str(Path(__file__).parents[2]),
        },
        encoding='utf-8',
        errors='replace',
    )

    assert result.returncode != 0
    # Matches the singular and the plural: how many objects the planted cycle
    # keeps alive is a VTK implementation detail.
    assert 'new VTK/plotter object' in result.stdout


@pytest.mark.needs_vtk_version(
    less_than=_VTK_GC_TRAVERSES_DICT,
    reason='This VTK collects the plotter; see test_plotter_collected_without_cleanup.',
)
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


@pytest.mark.needs_vtk_version(
    at_least=_VTK_GC_TRAVERSES_DICT,
    reason='This VTK cannot collect the plotter; see test_leak_pv_plotter.',
)
def test_plotter_collected_without_cleanup() -> None:
    """A plotter whose cleanup is disabled is still garbage-collectable."""

    def noop(*args: Any, **kwargs: Any):  # noqa: ARG001
        return None

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.enable_point_picking()
    pl.close = noop
    pl.deep_clean = noop
    ref = weakref.ref(pl)
    pv.close_all()  # drop the _ALL_PLOTTERS reference; close itself is a noop
    del pl
    gc.collect()
    assert ref() is None
