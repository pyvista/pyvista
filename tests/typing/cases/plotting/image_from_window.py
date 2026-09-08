"""Typing cases for :func:`pyvista.plotting.utilities.image_from_window`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from pyvista.core._typing_core import NumpyArray
from pyvista.plotting.utilities import image_from_window


def a_window() -> _vtk.vtkRenderWindow:
    """Return the render window of a plotter that has rendered one mesh."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.show(auto_close=False)
    assert pl.render_window is not None
    return pl.render_window


SKIP_RUNTIME = (
    dict.fromkeys(
        [
            'image_from_window(a_window())',
            'image_from_window(a_window(), as_vtk=False)',
            'image_from_window(a_window(), ignore_alpha=True, scale=2)',
            'image_from_window(a_window(), as_vtk=True)',
            'image_from_window(a_window(), as_vtk=a_flag())',
        ],
        'the VTK 9.3 wheel renders only through a display, and the core phase has none',
    )
    if pv.vtk_version_info < (9, 4)
    else {}
)


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(image_from_window(a_window()), NumpyArray[np.uint8])
assert_types(image_from_window(a_window(), as_vtk=False), NumpyArray[np.uint8])
assert_types(image_from_window(a_window(), ignore_alpha=True, scale=2), NumpyArray[np.uint8])
assert_types(image_from_window(a_window(), as_vtk=True), pv.ImageData)

# The catch-all, reached only by a flag widened to `bool`
assert_types(image_from_window(a_window(), as_vtk=a_flag()), NumpyArray[np.uint8] | pv.ImageData)
