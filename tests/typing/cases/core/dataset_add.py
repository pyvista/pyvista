"""Typing cases for :meth:`pyvista.DataSetFilters.__add__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured

# A merge gives an unstructured grid unless every input is a point cloud
assert_types(image() + poly(), pv.UnstructuredGrid)
assert_types(unstructured() + poly(), pv.UnstructuredGrid)
assert_types(unstructured() + [poly(), poly()], pv.UnstructuredGrid)  # noqa: RUF005
assert_types(image() + multiblock(), pv.UnstructuredGrid)
assert_types(pointset() + pointset(), pv.PointSet)
assert_types(pointset() + [pointset(), pointset()], pv.PointSet)  # noqa: RUF005
assert_types(pointset() + poly(), pv.UnstructuredGrid)

# Only the blocks decide what a composite merges into
assert_types(pointset() + multiblock(), pv.PointSet | pv.UnstructuredGrid)
