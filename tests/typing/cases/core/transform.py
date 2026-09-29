"""Typing cases for :meth:`pyvista.DataObjectFilters.transform`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import multiblock_poly
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured


def a_transform() -> pv.Transform:
    """Return a transform with a translation composed into it."""
    return pv.Transform().translate((1.0, 2.0, 3.0))


# Every dataset gives back its own class
assert_types(poly().transform(a_transform(), inplace=False), pv.PolyData)
assert_types(image().transform(a_transform(), inplace=False), pv.ImageData)
assert_types(rectilinear().transform(a_transform(), inplace=False), pv.RectilinearGrid)
assert_types(structured().transform(a_transform(), inplace=False), pv.StructuredGrid)
assert_types(unstructured().transform(a_transform(), inplace=False), pv.UnstructuredGrid)
assert_types(explicit_structured().transform(a_transform(), inplace=False), pv.ExplicitStructuredGrid)
assert_types(pointset().transform(a_transform(), inplace=False), pv.PointSet)
assert_types(multiblock().transform(a_transform(), inplace=False), pv.MultiBlock)

# A declared block type survives the transform
assert_types(multiblock_poly().transform(a_transform(), inplace=False), pv.MultiBlock[pv.PolyData])
