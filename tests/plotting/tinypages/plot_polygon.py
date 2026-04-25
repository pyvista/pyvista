from __future__ import annotations

import pyvista as pv


def not_called():
    """Never called if plot_directive works as expected."""
    raise NotImplementedError


def plot_poly():
    """The function that should be executed."""
    pv.Polygon().plot()
