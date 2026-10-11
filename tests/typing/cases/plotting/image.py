"""Typing cases for :attr:`pyvista.Plotter.image`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv


def a_plotter() -> pv.Plotter:
    """Return a plotter that has rendered one mesh and is still open."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.show(auto_close=False)
    return pl


with skip_runtime(pv.vtk_version_info < (9, 4), reason='the VTK 9.3 wheel renders only through a display'):
    assert_types(a_plotter().image, NDArray[np.uint8])
