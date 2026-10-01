"""Typing cases for :meth:`pyvista.DataObjectFilters.slice`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import multiblock_image
from tests.typing.meshes import multiblock_optional_poly
from tests.typing.meshes import multiblock_pointset
from tests.typing.meshes import multiblock_poly
from tests.typing.meshes import poly

# Slicing reduces any dataset to a surface, and a composite stays a composite
assert_types(poly().slice(), pv.PolyData)
assert_types(image().slice(), pv.PolyData)
assert_types(multiblock().slice(), pv.MultiBlock)

# A declared block type follows the filter through
assert_types(multiblock_poly().slice(), pv.MultiBlock[pv.PolyData])
assert_types(multiblock_image().slice(), pv.MultiBlock[pv.PolyData])

# An empty block survives the filter
assert_types(multiblock_optional_poly().slice(), pv.MultiBlock[pv.PolyData | None])

# A composite of point clouds cannot be reduced to a surface
assert_types(multiblock_pointset().slice(), Never)
