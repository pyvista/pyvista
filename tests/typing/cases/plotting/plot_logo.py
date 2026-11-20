"""Typing cases for :func:`pyvista.demos.plot_logo`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from type_assert import assert_types

import pyvista as pv
from pyvista import demos

if TYPE_CHECKING:
    from pyvista.plotting.plotter import _ShowReturnType
else:
    _ShowReturnType = pv.CameraPosition | None  # what an off-screen show returns


def a_flag() -> bool:
    """Return a flag whose value is not known statically."""
    return False


assert_types(demos.plot_logo(just_return_plotter=True), pv.Plotter)
assert_types(demos.plot_logo(), _ShowReturnType)
assert_types(demos.plot_logo(just_return_plotter=False), _ShowReturnType)
assert_types(demos.plot_logo(just_return_plotter=a_flag()), pv.Plotter | _ShowReturnType)
