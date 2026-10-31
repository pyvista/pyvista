"""Benchmarks for reflecting each mesh type, which reorders its cells to keep them oriented."""

from __future__ import annotations

import pytest

import pyvista as pv


def _image():
    """Return an image of 8000 voxels."""
    return pv.ImageData(dimensions=(21, 21, 21))


def _hexahedra():
    """Return an unstructured grid of 8000 hexahedra."""
    return _image().cast_to_structured_grid().cast_to_unstructured_grid()


MESHES = {
    'image': _image,
    'rectilinear': lambda: _image().cast_to_rectilinear_grid(),
    'structured': lambda: _image().cast_to_structured_grid(),
    'explicit': lambda: _image().cast_to_structured_grid().cast_to_explicit_structured_grid(),
    'voxels': lambda: _image().cast_to_unstructured_grid(),
    'hexahedra': _hexahedra,
    'quads': lambda: pv.Plane(i_resolution=90, j_resolution=90).cast_to_unstructured_grid(),
    'lines': lambda: pv.ImageData(dimensions=(8001, 1, 1)).cast_to_unstructured_grid(),
    'tetrahedra': lambda: _hexahedra().triangulate(),
    'polydata': lambda: pv.Sphere(theta_resolution=90, phi_resolution=90),
}

# Rectilinear grids and voxels reflect only across planes normal to an axis
OBLIQUE_MESHES = [name for name in MESHES if name not in ('rectilinear', 'voxels')]


@pytest.mark.parametrize('name', list(MESHES))
def test_reflect_axis(name, benchmark):
    """Reflect a mesh across a plane normal to an axis."""
    mesh = MESHES[name]()
    # Reflect once first, so the benchmarks' random order cannot move first-call costs
    mesh.reflect((1, 0, 0))
    assert benchmark(mesh.reflect, (1, 0, 0)).n_cells == mesh.n_cells


@pytest.mark.parametrize('name', OBLIQUE_MESHES)
def test_reflect_oblique(name, benchmark):
    """Reflect a mesh across a plane that is not normal to an axis."""
    mesh = MESHES[name]()
    # Reflect once first, so the benchmarks' random order cannot move first-call costs
    mesh.reflect((1, 2, 3))
    assert benchmark(mesh.reflect, (1, 2, 3)).n_cells == mesh.n_cells


def test_rotate_voxels_quarter_turn(benchmark):
    """Rotate voxels a quarter turn, which reorders their corners."""
    mesh = MESHES['voxels']()
    # Rotate once first, so the benchmarks' random order cannot move first-call costs
    mesh.rotate_z(90)
    assert benchmark(mesh.rotate_z, 90).n_cells == mesh.n_cells
