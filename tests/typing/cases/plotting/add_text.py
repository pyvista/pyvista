"""Typing cases for :meth:`pyvista.Plotter.add_text`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista.plotting.text import TextPositionOptions


def a_plotter() -> pv.Plotter:
    """Return a plotter to add text to."""
    return pv.Plotter()


def a_position() -> TextPositionOptions | Sequence[float] | None:
    """Return a position typed as everything `add_text` accepts, so the catch-all applies."""
    return 'upper_left'


# A named position is drawn as a corner annotation
assert_types(a_plotter().add_text('a'), pv.CornerAnnotation)
assert_types(a_plotter().add_text('a', position='upper_edge'), pv.CornerAnnotation)

# A coordinate, or no position at all, is drawn as free text
assert_types(a_plotter().add_text('a', position=(0.1, 0.1)), pv.Text)
assert_types(a_plotter().add_text('a', position=[0.1, 0.1], viewport=True), pv.Text)
assert_types(a_plotter().add_text('a', position=None), pv.Text)

# The catch-all, reached only by a position widened to the whole union
assert_types(a_plotter().add_text('a', position=a_position()), pv.CornerAnnotation | pv.Text)
