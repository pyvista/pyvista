"""Fixtures shared by the benchmarks.

Every fixture builds its data outside the measured region. Fixtures whose object a
benchmark mutates are function-scoped so one benchmark cannot perturb another.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk

pv.OFF_SCREEN = True

# Pin VTK to one thread so instruction counts do not depend on work splitting.
_vtk.vtkSMPTools.Initialize(1)


@pytest.fixture(scope='session')
def rng():
    """Return a seeded random generator."""
    return np.random.default_rng(0)


@pytest.fixture(scope='session')
def sphere(rng):
    """Return a small sphere carrying point scalars, vectors and normals."""
    mesh = pv.Sphere()
    mesh.point_data['data'] = rng.random(mesh.n_points)
    mesh.point_data['vec'] = rng.random((mesh.n_points, 3))
    mesh.set_active_scalars('data')
    return mesh


@pytest.fixture(scope='session')
def unchosen_sphere(rng):
    """Return a sphere whose active arrays are resolved per read, none having been chosen."""
    mesh = pv.Sphere()
    mesh.point_data['data'] = rng.random(mesh.n_points)
    mesh.point_data['vec'] = rng.random((mesh.n_points, 3))
    return mesh


@pytest.fixture(scope='session')
def big_sphere(rng):
    """Return a sphere large enough that array traffic dominates per-call overhead."""
    mesh = pv.Sphere(theta_resolution=200, phi_resolution=200)
    mesh.point_data['data'] = rng.random(mesh.n_points)
    return mesh


@pytest.fixture(scope='session')
def point_cloud(rng):
    """Return an array of unstructured points."""
    return rng.random((50_000, 3))


@pytest.fixture(scope='session')
def triangles():
    """Return a regular triangle connectivity array."""
    return np.arange(30_000).reshape(-1, 3)


@pytest.fixture(scope='session')
def triangle_points(rng):
    """Return points matching the regular triangle connectivity."""
    return rng.random((30_000, 3))


@pytest.fixture(scope='session')
def label_image(rng):
    """Return an image with integer cell labels."""
    image = pv.ImageData(dimensions=(60, 60, 60))
    image.cell_data['labels'] = rng.integers(0, 10, image.n_cells)
    return image


@pytest.fixture(scope='session')
def value_image(rng):
    """Return an image with int16 point values."""
    image = pv.ImageData(dimensions=(80, 80, 80))
    image.point_data['v'] = rng.integers(0, 100, image.n_points).astype(np.int16)
    return image


@pytest.fixture(scope='session')
def vtk_polydata():
    """Return an unwrapped VTK polydata."""
    poly = _vtk.vtkPolyData()
    poly.ShallowCopy(pv.Sphere())
    return poly


@pytest.fixture(scope='session')
def vtk_id_list():
    """Return a VTK id list holding a thousand ids."""
    ids = _vtk.vtkIdList()
    for i in range(1000):
        ids.InsertNextId(i)
    return ids


@pytest.fixture(scope='session')
def string_array():
    """Return a NumPy array of a thousand strings."""
    return np.array([f'label{i}' for i in range(1000)])


@pytest.fixture(scope='session')
def hex_grid():
    """Return an unstructured grid of hexahedra."""
    return pv.ImageData(dimensions=(10, 10, 10)).cast_to_unstructured_grid()


@pytest.fixture(scope='session')
def hex_cell(hex_grid):
    """Return a single hexahedral cell."""
    return hex_grid.get_cell(0)


@pytest.fixture(scope='session')
def triangle_cell(sphere):
    """Return a single triangular cell."""
    return sphere.get_cell(0)


@pytest.fixture(scope='session')
def ndarray(rng):
    """Return a pyvista_ndarray detached from any dataset."""
    return pv.pyvista_ndarray(rng.random(10_000))


@pytest.fixture(scope='session')
def dense_points(rng):
    """Return a large point array."""
    return rng.random((200_000, 3))


@pytest.fixture(scope='session')
def blocks():
    """Return a list of meshes for building a composite."""
    return [pv.Sphere() for _ in range(50)]


@pytest.fixture(scope='session')
def multiblock(blocks):
    """Return a composite of meshes."""
    return pv.MultiBlock(blocks)


@pytest.fixture
def mutable_sphere(rng):
    """Return a sphere a benchmark may write to."""
    mesh = pv.Sphere()
    mesh.point_data['data'] = rng.random(mesh.n_points)
    mesh.set_active_scalars('data')
    return mesh


@pytest.fixture
def scalars(sphere, rng):
    """Return an array sized to the small sphere's points."""
    return rng.random(sphere.n_points)
