"""Typing cases for :meth:`pyvista.DataObjectFilters.clip_box`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import multiblock_image
from tests.typing.meshes import multiblock_optional_image
from tests.typing.meshes import multiblock_optional_pointset
from tests.typing.meshes import multiblock_optional_poly
from tests.typing.meshes import multiblock_pointset
from tests.typing.meshes import multiblock_poly
from tests.typing.meshes import multiblock_unstructured
from tests.typing.meshes import pointset
from tests.typing.meshes import poly

# A box clip splits the cells it cuts, and clips a point cloud through its vertices
assert_types(poly().clip_box(), pv.PolyData)
assert_types(pointset().clip_box(), pv.PointSet)
assert_types(image().clip_box(), pv.UnstructuredGrid)
assert_types(multiblock().clip_box(), pv.MultiBlock)

# A declared block type follows the filter through
assert_types(multiblock_poly().clip_box(), pv.MultiBlock[pv.PolyData])
assert_types(multiblock_image().clip_box(), pv.MultiBlock[pv.UnstructuredGrid])
assert_types(multiblock_pointset().clip_box(), pv.MultiBlock[pv.PointSet])
assert_types(multiblock_unstructured().clip_box(), pv.MultiBlock[pv.UnstructuredGrid])

# An empty block survives the filter
assert_types(multiblock_optional_poly().clip_box(), pv.MultiBlock[pv.PolyData | None])
assert_types(multiblock_optional_image().clip_box(), pv.MultiBlock[pv.UnstructuredGrid | None])
assert_types(multiblock_optional_pointset().clip_box(), pv.MultiBlock[pv.PointSet | None])
