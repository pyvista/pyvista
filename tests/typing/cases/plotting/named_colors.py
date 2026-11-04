"""Typing cases for :data:`pyvista.typing.NamedColors` and :data:`pyvista.typing.NamedColormaps`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from pyvista.typing import ColorLike
    from pyvista.typing import ColormapOptions
    from pyvista.typing import NamedColormaps
    from pyvista.typing import NamedColors

color: NamedColors = 'tomato'
cmap: NamedColormaps = 'viridis'
color_like: ColorLike = color
cmap_options: ColormapOptions = cmap

assert_types(pv.Color(color), pv.Color)
assert_types(pv.LookupTable(cmap=cmap), pv.LookupTable)

unknown_color: NamedColors = 'not_a_color'  # type: ignore[assignment]
unknown_cmap: NamedColormaps = 'not_a_cmap'  # type: ignore[assignment]
