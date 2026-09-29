"""Typing cases for :meth:`pyvista.Table.update`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def float64_columns() -> dict[str, NDArray[np.float64]]:
    """Return float64 columns."""
    return {'a': np.zeros(3, dtype=np.float64)}


def int64_columns() -> dict[str, NDArray[np.int64]]:
    """Return int64 columns."""
    return {'b': np.zeros(3, dtype=np.int64)}


assert_types(pv.Table().update(float64_columns()), None)
assert_types(pv.Table().update(int64_columns()), None)
assert_types(pv.Table().update({'s': np.array(['x', 'y', 'z'])}), None)
assert_types(pv.Table().update([[0.0, 1.0], [2.0, 3.0]]), None)
assert_types(pv.Table().update(pv.Table().row_arrays), None)
