"""Typing cases for :meth:`pyvista.Transform.as_rotation`."""

from __future__ import annotations

from scipy.spatial.transform import Rotation
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray


def a_rotated_transform() -> pv.Transform:
    """Return a transform with a rotation composed into it."""
    return pv.Transform().rotate_z(30.0)


assert_types(a_rotated_transform().as_rotation(), Rotation)
assert_types(a_rotated_transform().as_rotation(None), Rotation)

assert_types(a_rotated_transform().as_rotation('quat'), NumpyArray[float])
assert_types(a_rotated_transform().as_rotation('matrix'), NumpyArray[float])
assert_types(a_rotated_transform().as_rotation('rotvec'), NumpyArray[float])
assert_types(a_rotated_transform().as_rotation('euler', 'xyz'), NumpyArray[float])
