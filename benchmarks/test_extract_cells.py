"""Benchmarks for extracting and removing cells and points from a surface."""

from __future__ import annotations

import numpy as np
import pytest

from pyvista.core.filters import data_set


@pytest.fixture(scope='session')
def label_surface(label_image):
    """Return the triangulated boundary surface between labels and a mask of one label's faces."""
    surface = label_image.contour_labels('all', scalars='labels', simplify_output=False)
    mask = np.any(surface.cell_data['boundary_labels'] == 0, axis=1)
    return surface, mask


@pytest.fixture(scope='session')
def label_surface_points(label_surface):
    """Return the label surface and a mask of the points on one side of its center."""
    surface, _ = label_surface
    return surface, surface.points[:, 0] > surface.center[0]


@pytest.fixture
def vtk_path(monkeypatch):
    """Send every mesh through the VTK extraction filters."""
    monkeypatch.setattr(data_set, '_homogeneous_cells', lambda _: None)


def test_remove_cells(label_surface, benchmark):
    """Remove faces from a surface with equal-sized faces."""
    surface, mask = label_surface
    assert benchmark(surface.remove_cells, mask).n_cells == surface.n_cells - mask.sum()


@pytest.mark.usefixtures('vtk_path')
def test_remove_cells_vtk(label_surface, benchmark):
    """Remove faces from a surface through the VTK extraction filter."""
    surface, mask = label_surface
    assert benchmark(surface.remove_cells, mask).n_cells == surface.n_cells - mask.sum()


def test_extract_cells(label_surface, benchmark):
    """Extract faces from a surface with equal-sized faces."""
    surface, mask = label_surface
    assert benchmark(surface.extract_cells, mask).n_cells == mask.sum()


@pytest.mark.usefixtures('vtk_path')
def test_extract_cells_vtk(label_surface, benchmark):
    """Extract faces from a surface through the VTK extraction filter."""
    surface, mask = label_surface
    assert benchmark(surface.extract_cells, mask).n_cells == mask.sum()


def test_remove_points(label_surface_points, benchmark):
    """Remove points from a surface with equal-sized faces."""
    surface, mask = label_surface_points
    assert benchmark(surface.remove_points, ind=mask).n_points < surface.n_points


@pytest.mark.usefixtures('vtk_path')
def test_remove_points_vtk(label_surface_points, benchmark):
    """Remove points from a surface through the VTK selection filter."""
    surface, mask = label_surface_points
    assert benchmark(surface.remove_points, ind=mask).n_points < surface.n_points


def test_extract_points(label_surface_points, benchmark):
    """Extract points from a surface with equal-sized faces."""
    surface, mask = label_surface_points
    assert benchmark(surface.extract_points, mask).n_points < surface.n_points


@pytest.mark.usefixtures('vtk_path')
def test_extract_points_vtk(label_surface_points, benchmark):
    """Extract points from a surface through the VTK selection filter."""
    surface, mask = label_surface_points
    assert benchmark(surface.extract_points, mask).n_points < surface.n_points
