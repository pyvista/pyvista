"""Typing cases for :meth:`pyvista.PolyDataFilters.merge`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def unstructured() -> pv.UnstructuredGrid:
    """Return a small unstructured grid."""
    return pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()


# Merging polydata stays polydata, and so does an in-place merge
assert_types(poly().merge(poly()), pv.PolyData)
assert_types(poly().merge([poly(), poly()]), pv.PolyData)
assert_types(poly().merge(poly(), inplace=True), pv.PolyData)
assert_types(poly().merge(unstructured()), pv.PolyData | pv.UnstructuredGrid)
assert_types(poly() + poly(), pv.PolyData)
assert_types(poly() + unstructured(), pv.PolyData | pv.UnstructuredGrid)
