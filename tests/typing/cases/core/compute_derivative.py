"""Typing cases for :meth:`pyvista.DataSetFilters.compute_derivative`."""

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

assert_types(with_arrays(poly()).compute_derivative(scalars='s'), pv.PolyData)
assert_types(with_arrays(image()).compute_derivative(scalars='s'), pv.ImageData)
assert_types(with_arrays(rectilinear()).compute_derivative(scalars='s'), pv.RectilinearGrid)
assert_types(with_arrays(structured()).compute_derivative(scalars='s'), pv.StructuredGrid)
assert_types(with_arrays(unstructured()).compute_derivative(scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).compute_derivative(scalars='s'), pv.ExplicitStructuredGrid)
assert_types(with_arrays(pointset()).compute_derivative(scalars='s'), pv.PointSet)
