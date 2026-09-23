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
            'a_plotter().screenshot(return_img=a_flag())',
        ],
        'the VTK 9.3 wheel renders only through a display, and the core phase has none',
    )
    if pv.vtk_version_info < (9, 4)
    else {}
)


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(a_plotter().screenshot(), NumpyArray[np.uint8])
assert_types(a_plotter().screenshot(return_img=True), NumpyArray[np.uint8])

assert_types(a_plotter().screenshot(return_img=False), None)

# The catch-all, reached only by a flag widened to `bool`
assert_types(a_plotter().screenshot(return_img=a_flag()), NumpyArray[np.uint8] | None)
