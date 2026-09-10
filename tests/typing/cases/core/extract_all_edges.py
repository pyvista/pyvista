"""Typing cases for :meth:`pyvista.DataObjectFilters.extract_all_edges`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

SKIP_RUNTIME = {
    'pointset().extract_all_edges()': 'a `PointSet` has no cells, so the call raises',
}


# Edges come back as a surface, whatever went in
assert_types(poly().extract_all_edges(), pv.PolyData)
assert_types(image().extract_all_edges(), pv.PolyData)
assert_types(rectilinear().extract_all_edges(), pv.PolyData)
assert_types(structured().extract_all_edges(), pv.PolyData)
assert_types(unstructured().extract_all_edges(), pv.PolyData)
assert_types(explicit_structured().extract_all_edges(), pv.PolyData)
assert_types(multiblock().extract_all_edges(), pv.MultiBlock)

assert_types(pointset().extract_all_edges(), Never)  # pragma: no cover
