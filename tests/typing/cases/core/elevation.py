"""Typing cases for :meth:`pyvista.DataObjectFilters.elevation`."""

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
assert_types(poly().elevation(), pv.PolyData)
assert_types(image().elevation(), pv.ImageData)
assert_types(rectilinear().elevation(), pv.RectilinearGrid)
assert_types(structured().elevation(), pv.StructuredGrid)
assert_types(unstructured().elevation(), pv.UnstructuredGrid)
assert_types(explicit_structured().elevation(), pv.ExplicitStructuredGrid)
assert_types(pointset().elevation(), pv.PointSet)
assert_types(multiblock().elevation(), pv.MultiBlock)
