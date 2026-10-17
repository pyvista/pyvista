"""Typing cases for :meth:`pyvista.DataSet.get_data_range`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

from tests.typing.meshes import poly

if TYPE_CHECKING:
    from numpy.typing import NDArray


def float64_values() -> NDArray[np.float64]:
    """Return float64 values."""
    return np.arange(3, dtype=np.float64)


assert_types(poly().get_data_range(), tuple[float, float])
assert_types(poly().get_data_range(float64_values()), tuple[float, float])
