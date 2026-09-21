"""Typing cases for :meth:`pyvista.DataSetFilters.sort_labels`."""

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

assert_types(with_arrays(poly()).sort_labels(scalars='labels'), pv.PolyData)
assert_types(with_arrays(image()).sort_labels(scalars='labels'), pv.ImageData)
assert_types(with_arrays(rectilinear()).sort_labels(scalars='labels'), pv.RectilinearGrid)
assert_types(with_arrays(structured()).sort_labels(scalars='labels'), pv.StructuredGrid)
assert_types(with_arrays(unstructured()).sort_labels(scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).sort_labels(scalars='labels'), pv.ExplicitStructuredGrid)
assert_types(with_arrays(pointset()).sort_labels(scalars='labels'), pv.PointSet)
