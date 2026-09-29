"""Typing cases for :meth:`pyvista.DataObjectFilters.clip_slab`."""

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

# A slab clip follows the plane clip
assert_types(poly().clip_slab(thickness=0.2, normal='z'), pv.PolyData)
assert_types(pointset().clip_slab(thickness=0.2, normal='z'), pv.PointSet)
assert_types(image().clip_slab(thickness=0.2, normal='z'), pv.UnstructuredGrid)
assert_types(multiblock().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock)

# A declared block type follows the filter through
assert_types(multiblock_poly().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock[pv.PolyData])
assert_types(multiblock_image().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock[pv.UnstructuredGrid])
assert_types(multiblock_pointset().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock[pv.PointSet])
assert_types(multiblock_unstructured().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock[pv.UnstructuredGrid])

# An empty block survives the filter
assert_types(multiblock_optional_poly().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock[pv.PolyData | None])
assert_types(multiblock_optional_image().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock[pv.UnstructuredGrid | None])
assert_types(multiblock_optional_pointset().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock[pv.PointSet | None])
