"""Typing cases for :meth:`pyvista.DataSetFilters.merge`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

assert_types(poly().merge(pv.Sphere()), pv.PolyData)
assert_types(image().merge(pv.Sphere()), pv.UnstructuredGrid)
assert_types(rectilinear().merge(pv.Sphere()), pv.UnstructuredGrid)
assert_types(structured().merge(pv.Sphere()), pv.UnstructuredGrid)
assert_types(unstructured().merge(pv.Sphere()), pv.UnstructuredGrid)
assert_types(explicit_structured().merge(pv.Sphere()), pv.UnstructuredGrid)
assert_types(pointset().merge(pv.Sphere()), pv.UnstructuredGrid)

# A PolyData stays a PolyData only when everything merged in is a PolyData
assert_types(poly().merge(image()), pv.UnstructuredGrid)
assert_types(poly().merge([pv.Sphere(), pv.Sphere()]), pv.PolyData)

# The blocks of a composite decide, and they are not known until it is merged
assert_types(poly().merge(multiblock()), pv.PolyData | pv.UnstructuredGrid)
assert_types(pointset().merge(multiblock()), pv.PointSet | pv.UnstructuredGrid)

assert_types(image().merge(), pv.UnstructuredGrid)
assert_types(rectilinear().merge(), pv.UnstructuredGrid)
assert_types(structured().merge(), pv.UnstructuredGrid)
assert_types(unstructured().merge(), pv.UnstructuredGrid)
assert_types(explicit_structured().merge(), pv.UnstructuredGrid)
assert_types(pointset().merge(), pv.PointSet)
