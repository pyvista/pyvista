"""Typing cases for :meth:`pyvista.DataSetMapper.set_scalars`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def a_mapper() -> pv.DataSetMapper:
    """Return a mapper of a sphere."""
    return pv.DataSetMapper(pv.Sphere())


def point_scalars() -> NDArray[np.float64]:
    """Return point scalars for a sphere."""
    return np.arange(pv.Sphere().n_points, dtype=np.float64)


def uint8_opacity() -> NDArray[np.uint8]:
    """Return a uint8 opacity transfer function."""
    return np.linspace(0, 255, 256).astype(np.uint8)


assert_types(a_mapper().set_scalars(point_scalars(), 'data', opacity=uint8_opacity()), None)
