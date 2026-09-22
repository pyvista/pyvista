"""Typing cases for :meth:`pyvista.LookupTable.map_value`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_lookup_table() -> pv.LookupTable:
    """Return a lookup table over the unit range."""
    return pv.LookupTable(cmap='viridis', scalar_range=(0.0, 1.0))


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(a_lookup_table().map_value(0.5), tuple[float, float, float, float])
assert_types(a_lookup_table().map_value(0.5, opacity=True), tuple[float, float, float, float])

assert_types(a_lookup_table().map_value(0.5, opacity=False), tuple[float, float, float])

# The catch-all, reached only by a flag widened to `bool`
assert_types(
    a_lookup_table().map_value(0.5, opacity=a_flag()),
    tuple[float, float, float] | tuple[float, float, float, float],
)
