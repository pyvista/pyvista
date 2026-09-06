"""Typing cases for :meth:`pyvista.DataSet.find_closest_point`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import VectorLike


def a_count() -> int:  # pragma: no cover
    """Return a count typed only as ``int``, so the second overload applies."""
    return 2


SKIP_RUNTIME = dict.fromkeys(
    [
        'pv.Sphere().find_closest_point((0.0, 1.0, 0.0), n=2)',
        'pv.Sphere().find_closest_point((0.0, 1.0, 0.0), n=a_count())',
    ],
    'an int64 array is not a `NumpyArray[int]` without the `npt_promote` mypy plugin',
)

# fmt: off

assert_types(pv.Sphere().find_closest_point((0.0, 1.0, 0.0)),               int)
assert_types(pv.Sphere().find_closest_point([0.0, 1.0, 0.0]),               int)
assert_types(pv.Sphere().find_closest_point(np.zeros(3)),                   int)
assert_types(pv.Sphere().find_closest_point((0.0, 1.0, 0.0), n=1),          int)

assert_types(pv.Sphere().find_closest_point((0.0, 1.0, 0.0), n=2),          VectorLike[int])  # pragma: no cover
assert_types(pv.Sphere().find_closest_point((0.0, 1.0, 0.0), n=a_count()),  VectorLike[int])  # pragma: no cover
