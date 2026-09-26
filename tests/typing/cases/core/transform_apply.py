"""Typing cases for :meth:`pyvista.Transform.apply`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def a_transform() -> pv.Transform:
    """Return a transform with a translation composed into it."""
    return pv.Transform().translate((1.0, 2.0, 3.0))


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding one mesh."""
    return pv.MultiBlock([pv.Sphere()])


def float32_points() -> NDArray[np.float32]:
    """Return float32 points."""
    return np.zeros((4, 3), dtype=np.float32)


def int32_points() -> NDArray[np.int32]:
    """Return int32 points."""
    return np.zeros((4, 3), dtype=np.int32)


def float32_rows() -> list[NDArray[np.float32]]:
    """Return a list of float32 rows."""
    return [np.zeros(3, dtype=np.float32)]


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return False


def mesh_points() -> pv.pyvista_ndarray:
    """Return the points of a mesh."""
    return pv.Sphere().points


# Arrays
def a_polydata_multiblock() -> pv.MultiBlock[pv.PolyData]:
    """Return a `MultiBlock` declared to hold only `PolyData`."""
    return pv.MultiBlock([pv.Sphere()])


assert_types(a_transform().apply(np.zeros((4, 3))), NDArray[np.float64])
assert_types(a_transform().apply([(0.0, 0.0, 0.0)]), NDArray[np.float64])
assert_types(a_transform().apply((0.0, 0.0, 0.0)), NDArray[np.float64])
assert_types(a_transform().apply(np.zeros((4, 3)), 'points'), NDArray[np.float64])
assert_types(a_transform().apply(np.zeros((4, 3)), 'vectors'), NDArray[np.float64])
assert_types(a_transform().apply(np.zeros((4, 3)), None), NDArray[np.float64])
assert_types(a_transform().apply(float32_points(), copy=False), NDArray[np.float32])
assert_types(a_transform().apply(float32_points(), 'vectors', copy=False), NDArray[np.float32])
assert_types(a_transform().apply(int32_points(), copy=False), NDArray[np.float64])
assert_types(a_transform().apply([(0.0, 0.0, 0.0)], copy=False), NDArray[np.float64])
assert_types(a_transform().apply(float32_points(), copy=a_flag()), NDArray[np.floating])
assert_types(a_transform().apply(float32_rows()), NDArray[np.float64])
assert_types(a_transform().apply(float32_rows(), copy=False), NDArray[np.floating])

# Datasets keep their own type
assert_types(a_transform().apply(pv.Sphere()), pv.PolyData)
assert_types(a_transform().apply(pv.ImageData(dimensions=(2, 2, 2))), pv.ImageData)
assert_types(a_transform().apply(a_multiblock()), pv.MultiBlock)
assert_types(a_transform().apply(pv.Sphere(), 'active_vectors'), pv.PolyData)
assert_types(a_transform().apply(pv.Sphere(), 'all_vectors'), pv.PolyData)
assert_types(a_transform().apply(pv.Sphere(), inverse=True), pv.PolyData)

# Actors widen to the base class the overload names
assert_types(a_transform().apply(pv.Actor()), pv.Prop3D)
assert_types(a_transform().apply(pv.Actor(), 'replace'), pv.Prop3D)
assert_types(a_transform().apply(pv.Actor(), 'pre-multiply'), pv.Prop3D)
assert_types(a_transform().apply(pv.Actor(), 'post-multiply'), pv.Prop3D)

# A declared block type survives the transform
assert_types(pv.Transform().apply(a_polydata_multiblock()), pv.MultiBlock[pv.PolyData])
assert_types(a_transform().apply(mesh_points()), NDArray[np.float64])
assert_types(a_transform().apply(mesh_points(), copy=False), NDArray[np.floating])
assert_types(a_transform().apply(mesh_points(), copy=a_flag()), NDArray[np.floating])
