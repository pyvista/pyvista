"""Typing cases for :meth:`pyvista.Plotter.add_volume`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def a_volume() -> pv.ImageData:
    """Return a grid with scalars to render as a volume."""
    grid = pv.ImageData(dimensions=(10, 10, 10))
    grid.point_data['scalars'] = np.linspace(0, 255, grid.n_points).astype(np.uint8)
    return grid


def a_plotter() -> pv.Plotter:
    """Return a plotter to add volumes to."""
    return pv.Plotter()


# A volume, not an actor, whatever went in
assert_types(a_plotter().add_volume(a_volume()), pv.Volume | list[pv.Volume])
assert_types(a_plotter().add_volume(np.zeros((10, 10, 10))), pv.Volume | list[pv.Volume])
assert_types(a_plotter().add_volume(pv.MultiBlock([a_volume()])), pv.Volume | list[pv.Volume])
