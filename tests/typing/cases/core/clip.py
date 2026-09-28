"""Typing cases for :meth:`pyvista.DataObjectFilters.clip`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import multiblock_dataset
from tests.typing.meshes import multiblock_image
from tests.typing.meshes import multiblock_optional_image
from tests.typing.meshes import multiblock_optional_pointset
from tests.typing.meshes import multiblock_optional_poly
from tests.typing.meshes import multiblock_pointset
from tests.typing.meshes import multiblock_poly
from tests.typing.meshes import multiblock_unstructured
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the widened overloads apply."""
    return True


# A plane clip keeps a surface a surface and a point cloud a point cloud
assert_types(poly().clip(), pv.PolyData)
assert_types(pointset().clip(), pv.PointSet)
assert_types(unstructured().clip(), pv.UnstructuredGrid)
assert_types(image().clip(), pv.UnstructuredGrid)
assert_types(multiblock().clip(), pv.MultiBlock)
# An ExplicitStructuredGrid is not an UnstructuredGrid, so it takes the DataSet overload
assert_types(explicit_structured().clip(), pv.UnstructuredGrid)

# `return_clipped` hands back both halves, of the same class
assert_types(poly().clip(return_clipped=True), tuple[pv.PolyData, pv.PolyData])
assert_types(pointset().clip(return_clipped=True), tuple[pv.PointSet, pv.PointSet])
assert_types(unstructured().clip(return_clipped=True), tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(image().clip(return_clipped=True), tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(multiblock().clip(return_clipped=True), tuple[pv.MultiBlock, pv.MultiBlock])

# A flag the caller computed gives both halves as one union
assert_types(poly().clip(return_clipped=a_flag()), pv.PolyData | tuple[pv.PolyData, pv.PolyData])
assert_types(pointset().clip(return_clipped=a_flag()), pv.PointSet | tuple[pv.PointSet, pv.PointSet])
assert_types(unstructured().clip(return_clipped=a_flag()), pv.UnstructuredGrid | tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(image().clip(return_clipped=a_flag()), pv.UnstructuredGrid | tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(multiblock().clip(return_clipped=a_flag()), pv.MultiBlock | tuple[pv.MultiBlock, pv.MultiBlock])

# Only the classes a clip can be copied back into accept `inplace`
assert_types(poly().clip(inplace=True), pv.PolyData)
assert_types(pointset().clip(inplace=True), pv.PointSet)
assert_types(unstructured().clip(inplace=True), pv.UnstructuredGrid)

# A declared block type follows the filter through
assert_types(multiblock_poly().clip(), pv.MultiBlock[pv.PolyData])
assert_types(multiblock_image().clip(), pv.MultiBlock[pv.UnstructuredGrid])
assert_types(multiblock_poly().clip(return_clipped=True), tuple[pv.MultiBlock[pv.PolyData], pv.MultiBlock[pv.PolyData]])
assert_types(multiblock_pointset().clip(), pv.MultiBlock[pv.PointSet])
assert_types(multiblock_unstructured().clip(), pv.MultiBlock[pv.UnstructuredGrid])

# An empty block survives the filter
assert_types(multiblock_optional_poly().clip(), pv.MultiBlock[pv.PolyData | None])
assert_types(multiblock_optional_image().clip(), pv.MultiBlock[pv.UnstructuredGrid | None])
assert_types(multiblock_optional_pointset().clip(), pv.MultiBlock[pv.PointSet | None])

# A block type declared only as `DataSet` takes the widest single-mesh return
assert_types(multiblock_dataset().clip(), pv.MultiBlock[pv.UnstructuredGrid])
