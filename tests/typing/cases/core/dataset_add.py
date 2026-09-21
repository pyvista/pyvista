"""Typing cases for :meth:`pyvista.DataSetFilters.__add__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured

# Adding to anything but a point cloud gives an unstructured grid
assert_types(image() + poly(), pv.UnstructuredGrid)
assert_types(unstructured() + poly(), pv.UnstructuredGrid)
assert_types(image() + [poly(), poly()], pv.UnstructuredGrid)  # noqa: RUF005
assert_types(image() + multiblock(), pv.UnstructuredGrid)

# A point cloud stays a point cloud only when everything added is one
assert_types(pointset() + pointset(), pv.PointSet)
assert_types(pointset() + [pointset(), pointset()], pv.PointSet)  # noqa: RUF005
assert_types(pointset() + poly(), pv.UnstructuredGrid)

# The blocks of a composite decide, and they are not known until it is merged
assert_types(pointset() + multiblock(), pv.PointSet | pv.UnstructuredGrid)
