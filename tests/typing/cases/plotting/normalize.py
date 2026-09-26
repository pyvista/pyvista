"""Typing cases for :func:`pyvista.plotting.tools.normalize`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from pyvista.plotting.tools import normalize


def float16_values() -> NDArray[np.float16]:
    """Return float16 values."""
    return np.arange(4, dtype=np.float16)


def float32_values() -> NDArray[np.float32]:
    """Return float32 values."""
    return np.arange(4, dtype=np.float32)


def float64_values() -> NDArray[np.float64]:
    """Return float64 values."""
    return np.arange(4, dtype=np.float64)


def int32_values() -> NDArray[np.int32]:
    """Return int32 values."""
    return np.arange(4, dtype=np.int32)


def floating_values() -> NDArray[np.floating]:
    """Return values of an unspecified floating dtype."""
    return np.arange(4, dtype=np.float64)


assert_types(normalize(np.arange(4)), NDArray[np.float64])
assert_types(normalize(float16_values()), NDArray[np.float16])
assert_types(normalize(float32_values(), minimum=0.0, maximum=3.0), NDArray[np.float32])
assert_types(normalize(float64_values()), NDArray[np.float64])
assert_types(normalize(int32_values(), minimum=0, maximum=3), NDArray[np.float64])
assert_types(normalize(floating_values()), NDArray[np.floating])
