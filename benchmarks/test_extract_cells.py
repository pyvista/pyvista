"""Benchmarks for extracting and removing cells from a surface."""

from __future__ import annotations

import numpy as np
import pytest

from pyvista.core.filters import data_set


@pytest.fixture(scope='session')
def label_surface(label_image):
    """Return the triangulated boundary surface between labels, with a mask of one label."""
    surface = label_image.contour_labels('all', scalars='labels', simplify_output=False)
    mask = np.any(surface.cell_data['boundary_labels'] == 0, axis=1)
    return surface, mask


@pytest.fixture
def vtk_path(monkeypatch):
    """Send regular-face surfaces through the VTK extraction filter."""
    monkeypatch.setattr(data_set, '_has_regular_faces_only', lambda _: False)


def test_remove_cells(label_surface, benchmark):
    """Remove faces from a surface with regular faces."""
    surface, mask = label_surface
    assert benchmark(surface.remove_cells, mask).n_cells == surface.n_cells - mask.sum()


@pytest.mark.usefixtures('vtk_path')
def test_remove_cells_vtk(label_surface, benchmark):
    """Remove faces from a surface through the VTK extraction filter."""
    surface, mask = label_surface
    assert benchmark(surface.remove_cells, mask).n_cells == surface.n_cells - mask.sum()


def test_extract_cells(label_surface, benchmark):
    """Extract faces from a surface with regular faces."""
    surface, mask = label_surface
    assert benchmark(surface.extract_cells, mask).n_cells == mask.sum()


@pytest.mark.usefixtures('vtk_path')
def test_extract_cells_vtk(label_surface, benchmark):
    """Extract faces from a surface through the VTK extraction filter."""
    surface, mask = label_surface
    assert benchmark(surface.extract_cells, mask).n_cells == mask.sum()
