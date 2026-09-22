"""Typing cases for :meth:`pyvista.DataSetFilters.gaussian_splatting`."""

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

assert_types(poly().gaussian_splatting(dimensions=(8, 8, 8)), pv.ImageData)
assert_types(image().gaussian_splatting(dimensions=(8, 8, 8)), pv.ImageData)
assert_types(rectilinear().gaussian_splatting(dimensions=(8, 8, 8)), pv.ImageData)
assert_types(structured().gaussian_splatting(dimensions=(8, 8, 8)), pv.ImageData)
assert_types(unstructured().gaussian_splatting(dimensions=(8, 8, 8)), pv.ImageData)
assert_types(explicit_structured().gaussian_splatting(dimensions=(8, 8, 8)), pv.ImageData)
assert_types(pointset().gaussian_splatting(dimensions=(8, 8, 8)), pv.ImageData)
