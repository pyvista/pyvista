"""Typing cases for :meth:`pyvista.Plotter.screenshot`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import pyvista_ndarray


def a_plotter() -> pv.Plotter:  # pragma: no cover
    """Return a plotter holding one mesh."""
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl


def a_flag() -> bool:  # pragma: no cover
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


SKIP_RUNTIME = dict.fromkeys(
    [
        'a_plotter().screenshot()',
        'a_plotter().screenshot(return_img=True)',
        'a_plotter().screenshot(return_img=False)',
        'a_plotter().screenshot(return_img=a_flag())',
    ],
    'the array `screenshot` returns is a plain `numpy.ndarray`, not the annotated `pyvista_ndarray`',
)

assert_types(a_plotter().screenshot(), pyvista_ndarray)  # pragma: no cover
assert_types(a_plotter().screenshot(return_img=True), pyvista_ndarray)  # pragma: no cover

assert_types(a_plotter().screenshot(return_img=False), None)  # pragma: no cover

# The catch-all, reached only by a flag widened to `bool`
assert_types(a_plotter().screenshot(return_img=a_flag()), pyvista_ndarray | None)  # pragma: no cover
