"""Typing cases for :meth:`pyvista.ImageDataFilters.crop`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import image

if TYPE_CHECKING:
    from numpy.typing import NDArray


def uint8_mask() -> NDArray[np.uint8]:
    """Return a uint8 label mask with one foreground point."""
    mask = np.zeros(image().n_points, dtype=np.uint8)
    mask[21] = 1
    return mask


def bool_mask() -> NDArray[np.bool_]:
    """Return a boolean mask with one foreground point."""
    return uint8_mask().astype(bool)


def list_mask() -> list[float]:
    """Return a mask as a list of floats with one foreground point."""
    return [float(value) for value in uint8_mask()]


assert_types(image().crop(mask=uint8_mask()), pv.ImageData)
assert_types(image().crop(mask=bool_mask()), pv.ImageData)
assert_types(image().crop(mask=list_mask()), pv.ImageData)
