"""Typing cases for :meth:`pyvista.DataSetFilters.warp_by_scalar`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured
from tests.typing.meshes import with_arrays

assert_types(with_arrays(poly()).warp_by_scalar(scalars='s'), pv.PolyData)
assert_types(with_arrays(image()).warp_by_scalar(scalars='s'), pv.StructuredGrid)
assert_types(with_arrays(rectilinear()).warp_by_scalar(scalars='s'), pv.StructuredGrid)
assert_types(with_arrays(structured()).warp_by_scalar(scalars='s'), pv.StructuredGrid)
assert_types(with_arrays(unstructured()).warp_by_scalar(scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).warp_by_scalar(scalars='s'), pv.ExplicitStructuredGrid)
assert_types(with_arrays(pointset()).warp_by_scalar(scalars='s'), pv.PointSet)
