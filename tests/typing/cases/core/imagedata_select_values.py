"""Typing cases for :meth:`pyvista.ImageDataFilters.select_values`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import image


def labeled_image() -> pv.ImageData:
    """Return an image carrying a label array."""
    mesh = image()
    mesh.point_data['labels'] = np.arange(mesh.n_points) % 3
    return mesh


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the widened overload applies."""
    return True


# Splitting the selection is what decides between an image and a composite
assert_types(labeled_image().select_values(1), pv.ImageData)
assert_types(labeled_image().select_values(1, split=False), pv.ImageData)
assert_types(labeled_image().select_values(1, split=True), pv.MultiBlock)
assert_types(labeled_image().select_values(ranges=[0, 1], split=True), pv.MultiBlock)
assert_types(labeled_image().select_values(1, split=a_flag()), pv.ImageData | pv.MultiBlock)
