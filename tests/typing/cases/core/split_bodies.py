"""Typing cases for :meth:`pyvista.DataSetFilters.split_bodies`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import examples


def as_data_set() -> pv.DataSet:
    """Return a grid typed only as the base class."""
    return examples.load_hexbeam()


def poly() -> pv.PolyData:
    """Return two disconnected spheres."""
    return pv.Sphere(center=(-4, 0, 0), phi_resolution=6, theta_resolution=6) + pv.Sphere(phi_resolution=6, theta_resolution=6)


# split_bodies is a MultiBlock whatever the input
assert_types(poly().split_bodies(), pv.MultiBlock)
assert_types(as_data_set().split_bodies(), pv.MultiBlock)
assert_types(examples.load_hexbeam().split_bodies(label=True), pv.MultiBlock)
