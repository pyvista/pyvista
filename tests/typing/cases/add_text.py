"""Typing cases for :meth:`pyvista.Plotter.add_text` and :meth:`~pyvista.Plotter.add_title`.

A named position gives a `CornerAnnotation` and a coordinate pair gives a `Text`.
"""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_plotter() -> pv.Plotter:
    """Return a plotter to add text to."""
    return pv.Plotter()


# fmt: off

assert_types(a_plotter().add_text('a'),                        pv.CornerAnnotation | pv.Text)
assert_types(a_plotter().add_text('a', position='upper_edge'), pv.CornerAnnotation | pv.Text)
assert_types(a_plotter().add_text('a', position=(0.1, 0.1)),   pv.CornerAnnotation | pv.Text)
assert_types(a_plotter().add_text('a', position=None),         pv.CornerAnnotation | pv.Text)

assert_types(a_plotter().add_title('a'),                       pv.CornerAnnotation | pv.Text)
