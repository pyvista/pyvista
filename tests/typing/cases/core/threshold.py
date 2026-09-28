"""Typing cases for :meth:`pyvista.DataSetFilters.threshold`."""

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

assert_types(with_arrays(poly()).threshold(0.0, scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(image()).threshold(0.0, scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(rectilinear()).threshold(0.0, scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(structured()).threshold(0.0, scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(unstructured()).threshold(0.0, scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).threshold(0.0, scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(pointset()).threshold(0.0, scalars='s'), pv.PointSet)
