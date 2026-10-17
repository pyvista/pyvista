"""Typing cases for :meth:`pyvista.DataSetAttributes.set_scalars`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

from tests.typing.meshes import poly

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_POINTS = poly().n_points


def complex_values() -> NDArray[np.complex128]:
    """Return one complex value per point."""
    return np.zeros(N_POINTS, dtype=np.complex128)


assert_types(poly().point_data.set_scalars([0.0] * N_POINTS), None)
assert_types(poly().point_data.set_scalars(complex_values(), 'c'), None)
