"""Typing cases for :attr:`pyvista.Plotter.image`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray


def a_plotter() -> pv.Plotter:
    """Return a plotter that has rendered one mesh and is still open."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.show(auto_close=False)
    return pl


assert_types(a_plotter().image, NumpyArray[np.uint8])
