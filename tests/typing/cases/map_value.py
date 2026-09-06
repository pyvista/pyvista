"""Typing cases for :meth:`pyvista.LookupTable.map_value`.

`opacity` decides whether the alpha channel is included, which the return type
states as a union of the two tuple lengths.
"""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv

_Color = tuple[float, float, float] | tuple[float, float, float, float]


def a_lookup_table() -> pv.LookupTable:
    """Return a lookup table over the unit range."""
    return pv.LookupTable(cmap='viridis', scalar_range=(0.0, 1.0))


assert_types(a_lookup_table().map_value(0.5), _Color)
assert_types(a_lookup_table().map_value(0.5, opacity=True), _Color)
assert_types(a_lookup_table().map_value(0.5, opacity=False), _Color)
