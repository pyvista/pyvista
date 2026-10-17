"""Typing cases for :func:`pyvista.demos.logo_letters`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import demos


def a_flag() -> bool:
    """Return a merge flag whose value is not a literal."""
    return pv.global_theme.notebook is None


assert_types(demos.logo_letters(), dict[str, pv.PolyData])
assert_types(demos.logo_letters(merge=False), dict[str, pv.PolyData])
assert_types(demos.logo_letters(merge=True), pv.PolyData)
assert_types(demos.logo_letters(merge=a_flag()), pv.PolyData | dict[str, pv.PolyData])
