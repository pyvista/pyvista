"""Typing cases for the filters on :class:`pyvista.DataObjectFilters`.

Each case names what the filter returns for that input class, taken from running it.
"""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def image() -> pv.ImageData:
    """Return a small uniform grid carrying point and cell data."""
    grid = pv.ImageData(dimensions=(5, 5, 5))
    grid.point_data['data'] = np.linspace(0.0, 1.0, grid.n_points)
    grid.cell_data['cells'] = np.linspace(0.0, 1.0, grid.n_cells)
    return grid


def rectilinear() -> pv.RectilinearGrid:
    """Return a small rectilinear grid."""
    axis = np.arange(5, dtype=float)
    return pv.RectilinearGrid(axis, axis, axis)


def unstructured() -> pv.UnstructuredGrid:
    """Return a tetrahedralized sphere."""
    return poly().delaunay_3d()


def pointset() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


def a_plane() -> _vtk.vtkPlane:
    """Return a plane through the origin."""
    plane = _vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)
    plane.SetNormal(0.0, 0.0, 1.0)
    return plane


def a_line() -> pv.PolyData:
    """Return a polyline crossing the test meshes."""
    return pv.Line((-1.0, -1.0, -1.0), (5.0, 5.0, 5.0), resolution=4)


# A slab clip keeps a surface a surface and tetrahedralizes the grids
assert_types(poly().clip_slab(normal='z', thickness=0.2), pv.PolyData)
assert_types(pointset().clip_slab(normal='z', thickness=0.2), pv.PointSet)
assert_types(multiblock().clip_slab(normal='z', thickness=0.2), pv.MultiBlock)
assert_types(image().clip_slab(normal='z', thickness=0.2), pv.UnstructuredGrid)

# A box clip tetrahedralizes even a surface
assert_types(poly().clip_box(), pv.UnstructuredGrid)
assert_types(image().clip_box(), pv.UnstructuredGrid)
assert_types(pointset().clip_box(), pv.PointSet)
assert_types(multiblock().clip_box(), pv.MultiBlock)

# Slicing reduces a dataset to a surface, and a composite stays a composite
assert_types(poly().slice(), pv.PolyData)
assert_types(image().slice(), pv.PolyData)
assert_types(rectilinear().slice(), pv.PolyData)
assert_types(multiblock().slice(), pv.MultiBlock)
assert_types(image().slice_implicit(a_plane()), pv.PolyData)
assert_types(multiblock().slice_implicit(a_plane()), pv.MultiBlock)
assert_types(image().slice_along_line(a_line()), pv.PolyData)
assert_types(multiblock().slice_along_line(a_line()), pv.MultiBlock)

# These two always collect their slices into a composite
assert_types(image().slice_orthogonal(), pv.MultiBlock)
assert_types(multiblock().slice_orthogonal(), pv.MultiBlock)
assert_types(image().slice_along_axis(n=2), pv.MultiBlock)
assert_types(multiblock().slice_along_axis(n=2), pv.MultiBlock)

assert_types(image().extract_all_edges(), pv.PolyData)
assert_types(multiblock().extract_all_edges(), pv.MultiBlock)

# Cell centres are points, whatever went in
assert_types(image().cell_centers(), pv.PolyData)
assert_types(pointset().cell_centers(), pv.PolyData)
assert_types(multiblock().cell_centers(), pv.MultiBlock)

# `PolyData` overrides `triangulate`, so only the grids reach this one
assert_types(image().triangulate(), pv.UnstructuredGrid)
assert_types(multiblock().triangulate(), pv.MultiBlock)

# These hand back the class they were given
assert_types(poly().elevation(), pv.PolyData)
assert_types(image().elevation(), pv.ImageData)
assert_types(pointset().elevation(), pv.PointSet)
assert_types(multiblock().elevation(), pv.MultiBlock)
assert_types(image().sample(image()), pv.ImageData)
assert_types(poly().sample(image()), pv.PolyData)
assert_types(image().compute_cell_sizes(), pv.ImageData)
assert_types(unstructured().compute_cell_sizes(), pv.UnstructuredGrid)
assert_types(image().cell_data_to_point_data(), pv.ImageData)
assert_types(image().ctp(), pv.ImageData)
assert_types(image().point_data_to_cell_data(), pv.ImageData)
assert_types(image().ptc(), pv.ImageData)
assert_types(image().cell_validator(), pv.ImageData)
assert_types(poly().cell_validator(), pv.PolyData)
