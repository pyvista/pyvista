"""Typing cases for :meth:`pyvista.DataSetFilters.remove_nan_cells`."""

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
from tests.typing.meshes import with_arrays

SKIP_RUNTIME = {
    "with_arrays(pointset()).remove_nan_cells(scalars='s')": 'a `PointSet` has no cells, so the call raises',
}


assert_types(with_arrays(poly()).remove_nan_cells(scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(image()).remove_nan_cells(scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(rectilinear()).remove_nan_cells(scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(structured()).remove_nan_cells(scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(unstructured()).remove_nan_cells(scalars='s'), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).remove_nan_cells(scalars='s'), pv.UnstructuredGrid)

assert_types(with_arrays(pointset()).remove_nan_cells(scalars='s'), Never)  # pragma: no cover
