"""Typing cases for :meth:`pyvista.Transform.as_rotation`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.spatial.transform import Rotation
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

if TYPE_CHECKING:
    from typing import Literal


def a_rotated_transform() -> pv.Transform:
    """Return a transform with a rotation composed into it."""
    return pv.Transform().rotate_z(30.0)


def a_representation() -> Literal['quat', 'matrix', 'rotvec', 'mrp', 'euler', 'davenport'] | None:
    """Return a representation typed as everything `as_rotation` accepts, so the catch-all applies."""
    return 'quat'


assert_types(a_rotated_transform().as_rotation(), Rotation)
assert_types(a_rotated_transform().as_rotation(None), Rotation)

assert_types(a_rotated_transform().as_rotation('quat'), NumpyArray[float])
assert_types(a_rotated_transform().as_rotation('matrix'), NumpyArray[float])
assert_types(a_rotated_transform().as_rotation('rotvec'), NumpyArray[float])
assert_types(a_rotated_transform().as_rotation('euler', 'xyz'), NumpyArray[float])

# The catch-all, reached only by a representation widened to the whole union
assert_types(a_rotated_transform().as_rotation(a_representation()), Rotation | NumpyArray[float])
