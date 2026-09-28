"""Typing cases for :meth:`pyvista.DataSetFilters.interpolate`."""

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

assert_types(poly().interpolate(pv.Sphere()), pv.PolyData)
assert_types(image().interpolate(pv.Sphere()), pv.ImageData)
assert_types(rectilinear().interpolate(pv.Sphere()), pv.RectilinearGrid)
assert_types(structured().interpolate(pv.Sphere()), pv.StructuredGrid)
assert_types(unstructured().interpolate(pv.Sphere()), pv.UnstructuredGrid)
assert_types(explicit_structured().interpolate(pv.Sphere()), pv.ExplicitStructuredGrid)
assert_types(pointset().interpolate(pv.Sphere()), pv.PointSet)
