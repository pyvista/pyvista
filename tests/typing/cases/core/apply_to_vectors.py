"""Typing cases for :meth:`pyvista.Transform.apply_to_vectors`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def float32_array() -> NDArray[np.float32]:
    """Return a float32 Nx3 array."""
    return np.zeros((2, 3), dtype=np.float32)


def float64_array() -> NDArray[np.float64]:
    """Return a float64 Nx3 array."""
    return np.zeros((2, 3))


def floating_array() -> NDArray[np.floating]:
    """Return an Nx3 array typed only as floating."""
    return np.zeros((2, 3), dtype=np.float32)


def int32_array() -> NDArray[np.int32]:
    """Return an int32 Nx3 array."""
    return np.zeros((2, 3), dtype=np.int32)


def float32_rows() -> list[NDArray[np.float32]]:
    """Return a list of float32 rows."""
    return [np.zeros(3, dtype=np.float32)]


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return False


def mesh_points() -> pv.pyvista_ndarray:
    """Return the points of a mesh."""
    return pv.Sphere().points


assert_types(pv.Transform().scale(2.0).apply_to_vectors(float32_array()), NDArray[np.float64])
assert_types(pv.Transform().scale(2.0).apply_to_vectors((1.0, 0.0, 0.0)), NDArray[np.float64])

# Without a copy, a floating array comes back as itself
assert_types(pv.Transform().scale(2.0).apply_to_vectors(float32_array(), copy=False), NDArray[np.float32])
assert_types(pv.Transform().scale(2.0).apply_to_vectors(float64_array(), copy=False), NDArray[np.float64])
assert_types(pv.Transform().scale(2.0).apply_to_vectors(floating_array(), copy=False), NDArray[np.floating])
assert_types(pv.Transform().scale(2.0).apply_to_vectors(int32_array(), copy=False), NDArray[np.float64])
assert_types(pv.Transform().scale(2.0).apply_to_vectors([[1, 2, 3]], copy=False), NDArray[np.float64])

# A sequence of arrays keeps a float32 dtype only without a copy
assert_types(pv.Transform().scale(2.0).apply_to_vectors(float32_rows()), NDArray[np.float64])
assert_types(pv.Transform().scale(2.0).apply_to_vectors(float32_rows(), copy=False), NDArray[np.floating])

# The catch-all, reached only by a flag widened to `bool`
assert_types(pv.Transform().scale(2.0).apply_to_vectors(float32_array(), copy=a_flag()), NDArray[np.floating])
assert_types(pv.Transform().apply_to_vectors(mesh_points(), copy=False), NDArray[np.floating])
assert_types(pv.Transform().apply_to_vectors(mesh_points(), copy=a_flag()), NDArray[np.floating])
