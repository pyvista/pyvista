"""Typing cases for :func:`pyvista.make_tri_mesh`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def float32_points() -> NDArray[np.float32]:
    """Return float32 triangle vertices."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def int64_points() -> NDArray[np.int64]:
    """Return int64 triangle vertices."""
    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64)


def int64_faces() -> NDArray[np.int64]:
    """Return int64 triangle indices."""
    return np.array([[0, 1, 2]], dtype=np.int64)


def uint32_faces() -> NDArray[np.uint32]:
    """Return uint32 triangle indices."""
    return np.array([[0, 1, 2]], dtype=np.uint32)


SKIP_RUNTIME = {
    'pv.make_tri_mesh(int64_points(), uint32_faces())': 'integer points warn that they are cast',
}

assert_types(pv.make_tri_mesh(float32_points(), int64_faces()), pv.PolyData)
assert_types(pv.make_tri_mesh(float32_points(), uint32_faces()), pv.PolyData)
assert_types(pv.make_tri_mesh(int64_points(), uint32_faces()), pv.PolyData)
