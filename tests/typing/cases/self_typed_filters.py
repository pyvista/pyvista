"""Typing cases for the filters that promise to hand back the type they were given."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core.filters.data_object import _MeshValidationReport


def a_grid() -> pv.ImageData:
    """Return a small grid."""
    return pv.ImageData(dimensions=(3, 3, 3))


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding one mesh."""
    return pv.MultiBlock([pv.Sphere()])


def some_contours() -> pv.PolyData:
    """Return a mesh made of lines."""
    return pv.Circle().extract_feature_edges()


# fmt: off

assert_types(pv.Sphere().cell_quality(),                                pv.PolyData)
assert_types(a_grid().cell_quality(),                                   pv.ImageData)
assert_types(a_multiblock().cell_quality(),                             pv.MultiBlock)
assert_types(pv.Sphere().cell_quality('area'),                          pv.PolyData)

assert_types(pv.Sphere().validate_mesh(),                               _MeshValidationReport[pv.PolyData])
assert_types(a_multiblock().validate_mesh(),                            _MeshValidationReport[pv.MultiBlock])

assert_types(a_grid().select_interior_points(pv.Sphere()),              pv.ImageData)
assert_types(pv.Sphere().select_interior_points(pv.Sphere()),           pv.PolyData)

assert_types(some_contours().triangulate_contours(),                    pv.PolyData)

assert_types(pv.Sphere().remove_unused_points(),                        pv.PolyData)
assert_types(pv.Sphere().cast_to_unstructured_grid().remove_unused_points(), pv.UnstructuredGrid)

assert_types(pv.Transform().apply_to_dataset(pv.Sphere()),              pv.PolyData)
assert_types(pv.Transform().apply_to_dataset(a_grid()),                 pv.ImageData)
assert_types(pv.Transform().apply_to_dataset(a_multiblock()),           pv.MultiBlock)
