"""Typing cases for :meth:`pyvista.DataObjectFilters.sample`."""

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

# Every dataset gives back its own class
assert_types(poly().sample(image()), pv.PolyData)
assert_types(image().sample(image()), pv.ImageData)
assert_types(rectilinear().sample(image()), pv.RectilinearGrid)
assert_types(structured().sample(image()), pv.StructuredGrid)
assert_types(unstructured().sample(image()), pv.UnstructuredGrid)
assert_types(explicit_structured().sample(image()), pv.ExplicitStructuredGrid)
assert_types(pointset().sample(image()), pv.PointSet)
assert_types(multiblock().sample(image()), pv.MultiBlock)
