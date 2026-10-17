"""Typing cases for :meth:`pyvista.DataSetFilters.extract_largest`."""

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


# extract_largest delegates to connectivity and keeps its return type
assert_types(poly().extract_largest(), pv.PolyData)
assert_types(point_set().extract_largest(), pv.PointSet)
assert_types(as_data_set().extract_largest(), pv.UnstructuredGrid)
assert_types(examples.load_hexbeam().extract_largest(), pv.UnstructuredGrid)
assert_types(poly().extract_largest(inplace=True), pv.PolyData)
assert_types(poly().extract_largest(progress_bar=True), pv.PolyData)
