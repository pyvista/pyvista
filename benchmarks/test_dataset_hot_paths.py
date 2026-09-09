"""Benchmarks for point access, copying and transforming a dataset."""

from __future__ import annotations

import operator

import numpy as np

import pyvista as pv
from pyvista.core.utilities.points import principal_axes


def _chained_transform():
    """Compose a translation, rotation and scale."""
    return pv.Transform().translate((1, 2, 3)).rotate_z(30).scale(2).matrix


def test_points_get(big_sphere, benchmark):
    """Read the points of a large mesh."""
    assert benchmark(operator.attrgetter('points'), big_sphere).size


def test_points_set(point_cloud, benchmark):
    """Assign points to a mesh."""
    mesh = pv.PolyData(point_cloud)
    benchmark(setattr, mesh, 'points', point_cloud)
    assert mesh.n_points == len(point_cloud)


def test_n_points(sphere, benchmark):
    """Read the point count."""
    assert benchmark(operator.attrgetter('n_points'), sphere)


def test_copy_deep(sphere, benchmark):
    """Deep-copy a dataset."""
    assert benchmark(sphere.copy, deep=True).n_points == sphere.n_points


def test_copy_meta_from(sphere, benchmark):
    """Copy active-attribute metadata onto another dataset."""
    target = pv.Sphere()
    benchmark(target.copy_meta_from, sphere, deep=True)
    assert target.n_points


def test_copy_from_shallow(sphere, benchmark):
    """Shallow-copy one dataset onto another."""
    target = pv.Sphere()
    benchmark(target.copy_from, sphere, deep=False)
    assert target.n_points == sphere.n_points


def test_transform(sphere, benchmark):
    """Transform a mesh by an identity matrix."""
    assert benchmark(sphere.transform, np.eye(4), inplace=False).n_points


def test_transform_chain(benchmark):
    """Compose a translation, rotation and scale into a matrix."""
    assert benchmark(_chained_transform).shape == (4, 4)


def test_dataset_equality(sphere, benchmark):
    """Compare two equal datasets."""
    other = sphere.copy()
    assert benchmark(sphere.__eq__, other)


def test_principal_axes(dense_points, benchmark):
    """Compute the principal axes of a large point set."""
    assert benchmark(principal_axes, dense_points).shape == (3, 3)
