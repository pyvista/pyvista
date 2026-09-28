"""Typing cases for :meth:`pyvista.DataSetFilters.voxelize_binary_mask`."""

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
    'pointset().voxelize_binary_mask(dimensions=(4, 4, 4))': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().voxelize_binary_mask(dimensions=(4, 4, 4)), pv.ImageData)
assert_types(image().voxelize_binary_mask(dimensions=(4, 4, 4)), pv.ImageData)
assert_types(rectilinear().voxelize_binary_mask(dimensions=(4, 4, 4)), pv.ImageData)
assert_types(structured().voxelize_binary_mask(dimensions=(4, 4, 4)), pv.ImageData)
assert_types(unstructured().voxelize_binary_mask(dimensions=(4, 4, 4)), pv.ImageData)
assert_types(explicit_structured().voxelize_binary_mask(dimensions=(4, 4, 4)), pv.ImageData)

assert_types(pointset().voxelize_binary_mask(dimensions=(4, 4, 4)), Never)  # pragma: no cover
