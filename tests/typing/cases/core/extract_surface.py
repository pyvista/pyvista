"""Typing cases for :meth:`pyvista.DataSetFilters.extract_surface`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

SKIP_RUNTIME = {
    "pointset().extract_surface(algorithm='dataset_surface')": ('a `PointSet` has no cells, so the call raises'),
}


# Every dataset with cells yields a surface
assert_types(poly().extract_surface(algorithm='dataset_surface'), pv.PolyData)
assert_types(image().extract_surface(algorithm='dataset_surface'), pv.PolyData)
assert_types(rectilinear().extract_surface(algorithm='dataset_surface'), pv.PolyData)
assert_types(structured().extract_surface(algorithm='dataset_surface'), pv.PolyData)
assert_types(unstructured().extract_surface(algorithm='dataset_surface'), pv.PolyData)
assert_types(explicit_structured().extract_surface(algorithm='dataset_surface'), pv.PolyData)

assert_types(pointset().extract_surface(algorithm='dataset_surface'), Never)  # pragma: no cover
