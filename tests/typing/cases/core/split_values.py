"""Typing cases for :meth:`pyvista.DataSetFilters.split_values`."""

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
from tests.typing.meshes import with_arrays

assert_types(with_arrays(poly()).split_values([0, 1], scalars='labels'), pv.MultiBlock)
assert_types(with_arrays(image()).split_values([0, 1], scalars='labels'), pv.MultiBlock)
assert_types(with_arrays(rectilinear()).split_values([0, 1], scalars='labels'), pv.MultiBlock)
assert_types(with_arrays(structured()).split_values([0, 1], scalars='labels'), pv.MultiBlock)
assert_types(with_arrays(unstructured()).split_values([0, 1], scalars='labels'), pv.MultiBlock)
assert_types(with_arrays(explicit_structured()).split_values([0, 1], scalars='labels'), pv.MultiBlock)
assert_types(with_arrays(pointset()).split_values([0, 1], scalars='labels'), pv.MultiBlock)
