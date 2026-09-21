"""Typing cases for :meth:`pyvista.DataSetFilters.__add__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured

# A merge gives one of the three classes it can build, never a bare dataset
assert_types(image() + poly(), pv.PolyData | pv.PointSet | pv.UnstructuredGrid)
assert_types(unstructured() + poly(), pv.PolyData | pv.PointSet | pv.UnstructuredGrid)
assert_types(unstructured() + [poly(), poly()], pv.PolyData | pv.PointSet | pv.UnstructuredGrid)  # noqa: RUF005
assert_types(pointset() + pointset(), pv.PolyData | pv.PointSet | pv.UnstructuredGrid)
assert_types(image() + multiblock(), pv.PolyData | pv.PointSet | pv.UnstructuredGrid)
