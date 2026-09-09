"""Typing cases for :meth:`pyvista.Plotter.screenshot`."""

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


SKIP_RUNTIME = (
    dict.fromkeys(
        [
            'a_plotter().screenshot()',
            'a_plotter().screenshot(return_img=True)',
            'a_plotter().screenshot(return_img=False)',
        ],
        'the VTK 9.3 wheel renders only through a display',
    )
    if pv.vtk_version_info < (9, 4)
    else {}
)

# A plain array of pixels, or nothing
assert_types(a_plotter().screenshot(), NumpyArray[np.uint8] | None)
assert_types(a_plotter().screenshot(return_img=True), NumpyArray[np.uint8] | None)
assert_types(a_plotter().screenshot(return_img=False), NumpyArray[np.uint8] | None)
