"""Benchmarks for extracting and removing cells and points from meshes with equal-sized cells."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.core.filters import data_set

KINDS = ['verts', 'lines', 'triangles', 'hexahedra']


@pytest.fixture(scope='session')
def homogeneous_meshes(label_image, rng):
    """Return one mesh per kind of equal-sized cell, each with about 100k cells or more."""
    n = 100_000
    segments = np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + 1])
    return {
        'verts': pv.PolyData(rng.random((n, 3))),
        'lines': pv.PolyData(rng.random((n + 1, 3)), lines=segments),
        'triangles': label_image.contour_labels('all', scalars='labels', simplify_output=False),
        'hexahedra': pv.ImageData(dimensions=(47, 47, 47)).cast_to_unstructured_grid(),
    }


@pytest.fixture(params=KINDS)
def homogeneous_mesh(request, homogeneous_meshes):
    """Return a mesh, a mask of every other cell and a mask of the points on one side."""
    mesh = homogeneous_meshes[request.param]
    cell_mask = np.arange(mesh.n_cells) % 2 == 0
    point_mask = mesh.points[:, 0] > mesh.center[0]
    return mesh, cell_mask, point_mask


@pytest.fixture(params=['numpy', 'vtk'])
def extraction_path(request, monkeypatch):
    """Take the NumPy fast path, or send every mesh through the VTK extraction filters."""
    if request.param == 'vtk':
        monkeypatch.setattr(data_set, '_homogeneous_cells', lambda _: None)


@pytest.mark.usefixtures('extraction_path')
def test_extract_cells(homogeneous_mesh, benchmark):
    """Extract every other cell."""
    mesh, cell_mask, _ = homogeneous_mesh
    assert benchmark(mesh.extract_cells, cell_mask).n_cells == cell_mask.sum()


@pytest.mark.usefixtures('extraction_path')
def test_remove_cells(homogeneous_mesh, benchmark):
    """Remove every other cell."""
    mesh, cell_mask, _ = homogeneous_mesh
    assert benchmark(mesh.remove_cells, cell_mask).n_cells == mesh.n_cells - cell_mask.sum()


@pytest.mark.usefixtures('extraction_path')
def test_extract_points(homogeneous_mesh, benchmark):
    """Extract the points on one side of the mesh with their cells."""
    mesh, _, point_mask = homogeneous_mesh
    if not isinstance(mesh, pv.PolyData):
        pytest.skip('point filters take the VTK path for grids')
    assert benchmark(mesh.extract_points, point_mask).n_points < mesh.n_points


@pytest.mark.usefixtures('extraction_path')
def test_remove_points(homogeneous_mesh, benchmark):
    """Remove the points on one side of the mesh with their cells."""
    mesh, _, point_mask = homogeneous_mesh
    if not isinstance(mesh, pv.PolyData):
        pytest.skip('point filters take the VTK path for grids')
    assert benchmark(mesh.remove_points, ind=point_mask).n_points < mesh.n_points
