"""Typing cases for :func:`pyvista.compare_images`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def uint8_image() -> NDArray[np.uint8]:
    """Return a black RGB uint8 image."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


assert_types(pv.compare_images(uint8_image(), uint8_image()), float)
assert_types(pv.compare_images(uint8_image(), uint8_image(), use_vtk=False), float)
