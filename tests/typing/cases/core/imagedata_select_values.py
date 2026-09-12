"""Typing cases for :meth:`pyvista.ImageDataFilters.select_values`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def image() -> pv.ImageData:
    """Return a labeled image."""
    mesh = pv.ImageData(dimensions=(4, 4, 4))
    mesh.point_data['labels'] = np.arange(mesh.n_points) % 3
    return mesh


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the widened overload applies."""
    return True


# Splitting the selection is what decides between an image and a composite
assert_types(image().select_values(1), pv.ImageData)
assert_types(image().select_values(1, split=False), pv.ImageData)
assert_types(image().select_values(1, split=True), pv.MultiBlock)
assert_types(image().select_values(ranges=[0, 1], split=True), pv.MultiBlock)
assert_types(image().select_values(1, split=a_flag()), pv.ImageData | pv.MultiBlock)
