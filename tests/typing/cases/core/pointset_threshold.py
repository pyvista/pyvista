"""Typing cases for :meth:`pyvista.PointSet.threshold`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_cloud() -> pv.PointSet:
    """Return a point cloud carrying scalars."""
    cloud = pv.PointSet(pv.Sphere().points)
    cloud.point_data['height'] = cloud.points[:, 2]
    return cloud


assert_types(a_cloud().threshold(0.0), pv.PointSet)
assert_types(a_cloud().threshold([-0.2, 0.2], scalars='height'), pv.PointSet)
