"""Typing cases for :attr:`pyvista.Volume.prop`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def a_volume() -> pv.Volume:
    """Return a volume added to a plotter, which is what wraps its property."""
    grid = pv.ImageData(dimensions=(10, 10, 10))
    grid.point_data['scalars'] = np.linspace(0, 255, grid.n_points).astype(np.uint8)
    return pv.Plotter().add_volume(grid)


assert_types(a_volume().prop, pv.VolumeProperty)
assert_types(a_volume().prop.copy(), pv.VolumeProperty)
