"""Typing cases for :meth:`pyvista.DataSetFilters.connectivity` and its callers."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import examples


def as_data_set() -> pv.DataSet:
    """Return a grid typed only as the base class."""
    return examples.load_hexbeam()


def poly() -> pv.PolyData:
    """Return two disconnected spheres."""
    return pv.Sphere(center=(-4, 0, 0), phi_resolution=6, theta_resolution=6) + pv.Sphere(phi_resolution=6, theta_resolution=6)


def point_set() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(np.random.default_rng(0).random((10, 3)))


assert_types(poly().connectivity(), pv.PolyData)
assert_types(point_set().connectivity(), pv.PointSet)
assert_types(as_data_set().connectivity(), pv.UnstructuredGrid)
assert_types(examples.load_hexbeam().connectivity(), pv.UnstructuredGrid)
assert_types(examples.load_uniform().connectivity(), pv.UnstructuredGrid)
assert_types(examples.load_rectilinear().connectivity(), pv.UnstructuredGrid)
assert_types(examples.load_structured().connectivity(), pv.UnstructuredGrid)
assert_types(examples.load_explicit_structured().connectivity(), pv.UnstructuredGrid)

assert_types(poly().connectivity('all'), pv.PolyData)
assert_types(poly().connectivity('largest'), pv.PolyData)
assert_types(poly().connectivity('specified', 0), pv.PolyData)
assert_types(poly().connectivity('cell_seed', 0), pv.PolyData)
assert_types(poly().connectivity('point_seed', 0), pv.PolyData)
assert_types(poly().connectivity('closest', (0.0, 0.0, 0.0)), pv.PolyData)

assert_types(poly().connectivity(extraction_mode='specified', region_ids=0), pv.PolyData)
assert_types(poly().connectivity(extraction_mode='specified', region_ids=[0, 1]), pv.PolyData)
assert_types(poly().connectivity(extraction_mode='cell_seed', cell_ids=0), pv.PolyData)
assert_types(poly().connectivity(extraction_mode='cell_seed', cell_ids=[0, 1]), pv.PolyData)
assert_types(
    poly().connectivity(extraction_mode='cell_seed', cell_ids=np.zeros(96, dtype=bool)),
    pv.PolyData,
)
assert_types(poly().connectivity(extraction_mode='point_seed', point_ids=0), pv.PolyData)
assert_types(poly().connectivity(extraction_mode='point_seed', point_ids=[0, 1]), pv.PolyData)
assert_types(
    poly().connectivity(extraction_mode='point_seed', point_ids=np.zeros(52, dtype=bool)),
    pv.PolyData,
)
assert_types(poly().connectivity(extraction_mode='closest', closest_point=(0.0, 0.0, 0.0)), pv.PolyData)
assert_types(poly().connectivity(extraction_mode='closest', closest_point=np.zeros(3)), pv.PolyData)

assert_types(poly().connectivity(scalar_range=(-1.0, 1.0)), pv.PolyData)
assert_types(poly().connectivity(scalar_range=[-1.0, 1.0], scalars='Normals'), pv.PolyData)
assert_types(poly().connectivity(scalar_range=np.array([-1.0, 1.0])), pv.PolyData)
assert_types(poly().connectivity(label_regions=True), pv.PolyData)
assert_types(poly().connectivity(label_regions=False), pv.PolyData)
assert_types(poly().connectivity(region_assignment_mode='ascending'), pv.PolyData)
assert_types(poly().connectivity(region_assignment_mode='descending'), pv.PolyData)
assert_types(poly().connectivity(region_assignment_mode='unspecified'), pv.PolyData)
assert_types(poly().connectivity(inplace=True), pv.PolyData)
assert_types(poly().connectivity(inplace=False), pv.PolyData)
assert_types(poly().connectivity(progress_bar=True), pv.PolyData)

assert_types(as_data_set().connectivity('largest', label_regions=False), pv.UnstructuredGrid)
assert_types(point_set().connectivity('all', inplace=True), pv.PointSet)

assert_types(poly().extract_largest(), pv.PolyData)
assert_types(point_set().extract_largest(), pv.PointSet)
assert_types(as_data_set().extract_largest(), pv.UnstructuredGrid)
assert_types(examples.load_hexbeam().extract_largest(), pv.UnstructuredGrid)
assert_types(poly().extract_largest(inplace=True), pv.PolyData)
assert_types(poly().extract_largest(progress_bar=True), pv.PolyData)

assert_types(poly().split_bodies(), pv.MultiBlock)
assert_types(as_data_set().split_bodies(), pv.MultiBlock)
assert_types(examples.load_hexbeam().split_bodies(label=True), pv.MultiBlock)
