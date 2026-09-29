"""Typing cases for :meth:`pyvista.Plotter.add_volume`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def a_volume() -> pv.ImageData:
    """Return a grid with scalars to render as a volume."""
    grid = pv.ImageData(dimensions=(10, 10, 10))
    grid.point_data['scalars'] = np.linspace(0, 255, grid.n_points).astype(np.uint8)
    return grid


def a_plotter() -> pv.Plotter:
    """Return a plotter to add volumes to."""
    return pv.Plotter()


def uint8_volume() -> NDArray[np.uint8]:
    """Return a uint8 volume array."""
    return np.zeros((10, 10, 10), dtype=np.uint8)


def int16_volume() -> NDArray[np.int16]:
    """Return an int16 volume array."""
    return np.zeros((10, 10, 10), dtype=np.int16)


def a_volume_or_composite() -> pv.DataSet | pv.MultiBlock:
    """Return a volume typed as either kind of input, so the catch-all overload applies."""
    return a_volume()


assert_types(a_plotter().add_volume(a_volume()), pv.Volume)
assert_types(a_plotter().add_volume(np.zeros((10, 10, 10))), pv.Volume)
assert_types(a_plotter().add_volume(uint8_volume()), pv.Volume)
assert_types(a_plotter().add_volume(int16_volume()), pv.Volume)
assert_types(a_plotter().add_volume(a_volume(), opacity=0.5), pv.Volume)
assert_types(a_plotter().add_volume(a_volume(), opacity=[0.0, 1.0]), pv.Volume)

# A composite gets one volume per block
assert_types(a_plotter().add_volume(pv.MultiBlock([a_volume()])), list[pv.Volume])

# The catch-all, reached only by an input widened to both kinds
assert_types(a_plotter().add_volume(a_volume_or_composite()), pv.Volume | list[pv.Volume])
