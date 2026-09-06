"""Typing cases for :meth:`pyvista.Plotter.add_text` and :meth:`~pyvista.Plotter.add_title`.

A named position gives a `CornerAnnotation` and a coordinate pair gives a `Text`,
which the return type states as a union rather than deciding on the argument.
"""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.plotting.text import CornerAnnotation
from pyvista.plotting.text import Text


def a_plotter() -> pv.Plotter:
    """Return a plotter to add text to."""
    return pv.Plotter()


# fmt: off

assert_types(a_plotter().add_text('a'),                              CornerAnnotation | Text)
assert_types(a_plotter().add_text('a', position='lower_left'),       CornerAnnotation | Text)
assert_types(a_plotter().add_text('a', position='upper_edge'),       CornerAnnotation | Text)
assert_types(a_plotter().add_text('a', position=(0.1, 0.1)),         CornerAnnotation | Text)
assert_types(a_plotter().add_text('a', position=None),               CornerAnnotation | Text)
assert_types(a_plotter().add_text('a', font_size=12),                CornerAnnotation | Text)
assert_types(a_plotter().add_text('a', color='red', shadow=True),    CornerAnnotation | Text)

assert_types(a_plotter().add_title('a'),                             CornerAnnotation | Text)
assert_types(a_plotter().add_title('a', font_size=12),               CornerAnnotation | Text)
