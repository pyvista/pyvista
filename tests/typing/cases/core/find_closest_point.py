"""Typing cases for :meth:`pyvista.DataSet.find_closest_point`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def a_count() -> int:
    """Return a count typed only as ``int``, so the second overload applies."""
    return 2


assert_types(pv.Sphere().find_closest_point((0.0, 1.0, 0.0)), int)
assert_types(pv.Sphere().find_closest_point([0.0, 1.0, 0.0]), int)
assert_types(pv.Sphere().find_closest_point(np.zeros(3)), int)
assert_types(pv.Sphere().find_closest_point((0.0, 1.0, 0.0), n=1), int)

assert_types(pv.Sphere().find_closest_point((0.0, 1.0, 0.0), n=2), NDArray[np.int_])
assert_types(pv.Sphere().find_closest_point((0.0, 1.0, 0.0), n=a_count()), NDArray[np.int_])
