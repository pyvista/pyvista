"""Typing cases for the filters on the composite and structured grid classes."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding two meshes."""
    return pv.MultiBlock([pv.Sphere(), pv.Cube()])


def a_rectilinear() -> pv.RectilinearGrid:
    """Return a small rectilinear grid."""
    axis = np.arange(4, dtype=float)
    return pv.RectilinearGrid(axis, axis, axis)


def a_structured() -> pv.StructuredGrid:
    """Return a small structured grid."""
    x, y, z = np.meshgrid(np.arange(4.0), np.arange(4.0), np.arange(4.0), indexing='ij')
    return pv.StructuredGrid(x, y, z)


def a_neighbouring_structured() -> pv.StructuredGrid:
    """Return a structured grid sharing its lower face with `a_structured`."""
    grid = a_structured()
    grid.points[:, 2] += 3.0
    return grid


def a_grid() -> pv.UnstructuredGrid:
    """Return a tetrahedralized sphere."""
    return pv.Sphere().delaunay_3d().cast_to_unstructured_grid()


SKIP_RUNTIME = {
    'a_multiblock().extract_geometry()': 'deprecated in favour of `extract_surface`',
}

assert_types(a_multiblock().extract_geometry(), pv.PolyData)  # pragma: no cover
assert_types(a_multiblock().combine(), pv.UnstructuredGrid)
assert_types(a_multiblock().combine(merge_points=True), pv.UnstructuredGrid)
assert_types(a_multiblock().outline(), pv.PolyData)
assert_types(a_multiblock().outline(generate_faces=True), pv.PolyData)
assert_types(a_multiblock().outline_corners(), pv.PolyData)

assert_types(a_rectilinear().to_tetrahedra(), pv.UnstructuredGrid)
assert_types(a_rectilinear().to_tetrahedra(tetra_per_cell=6), pv.UnstructuredGrid)

assert_types(a_structured().extract_subset((0, 2, 0, 2, 0, 2)), pv.StructuredGrid)
assert_types(a_structured().concatenate(a_neighbouring_structured(), axis=2), pv.StructuredGrid)

assert_types(a_grid().subdivide_tetra(), pv.UnstructuredGrid)
assert_types(a_grid().clean(), pv.UnstructuredGrid)
assert_types(a_grid().clean(tolerance=0.01), pv.UnstructuredGrid)
