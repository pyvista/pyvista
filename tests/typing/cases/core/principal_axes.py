"""Typing cases for :func:`pyvista.principal_axes`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def some_points() -> NDArray[np.floating]:
    """Return points with a distinct variance along each axis."""
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])


def float32_points() -> NDArray[np.float32]:
    """Return float32 points."""
    return some_points().astype(np.float32)


def float64_points() -> NDArray[np.float64]:
    """Return float64 points."""
    return some_points().astype(np.float64)


def int32_points() -> NDArray[np.int32]:
    """Return int32 points."""
    return some_points().astype(np.int32)


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


def mesh_points() -> pv.pyvista_ndarray:
    """Return the points of a mesh."""
    return pv.Sphere().points


_Pair = tuple[NDArray[np.floating], NDArray[np.floating]]
_Pair32 = tuple[NDArray[np.float32], NDArray[np.float32]]
_Pair64 = tuple[NDArray[np.float64], NDArray[np.float64]]
_LIST = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]

assert_types(pv.principal_axes(some_points()), NDArray[np.floating])
assert_types(pv.principal_axes(some_points(), return_std=False), NDArray[np.floating])
assert_types(pv.principal_axes(some_points(), return_std=True), _Pair)

# Floating arrays keep their dtype
assert_types(pv.principal_axes(float32_points()), NDArray[np.float32])
assert_types(pv.principal_axes(float32_points(), return_std=True), _Pair32)
assert_types(pv.principal_axes(float64_points()), NDArray[np.float64])
assert_types(pv.principal_axes(float64_points(), return_std=True), _Pair64)

# Integer arrays and nested lists give float64
assert_types(pv.principal_axes(int32_points()), NDArray[np.float64])
assert_types(pv.principal_axes(int32_points(), return_std=True), _Pair64)
assert_types(pv.principal_axes(_LIST), NDArray[np.float64])
assert_types(pv.principal_axes(_LIST, return_std=True), _Pair64)

# The catch-alls, reached only by a flag widened to `bool`
assert_types(pv.principal_axes(some_points(), return_std=a_flag()), NDArray[np.floating] | _Pair)
assert_types(pv.principal_axes(_LIST, return_std=a_flag()), NDArray[np.floating] | _Pair)
assert_types(pv.principal_axes(mesh_points()), NDArray[np.floating])
assert_types(pv.principal_axes(mesh_points(), return_std=True), tuple[NDArray[np.floating], NDArray[np.floating]])
