"""Typing cases for :meth:`pyvista.DataSetFilters.clip_scalar`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def pointset() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25), origin=(-0.5, -0.5, -0.5))


def unstructured() -> pv.UnstructuredGrid:
    """Return a small unstructured grid."""
    return image().cast_to_unstructured_grid()


def poly_with_scalars() -> pv.PolyData:
    """Return a sphere carrying point scalars."""
    mesh = poly()
    mesh.point_data['data'] = mesh.points[:, 2]
    return mesh


def image_with_scalars() -> pv.ImageData:
    """Return a uniform grid carrying point scalars."""
    mesh = image()
    mesh.point_data['data'] = np.linspace(0.0, 1.0, mesh.n_points)
    return mesh


def pointset_with_scalars() -> pv.PointSet:
    """Return a point cloud carrying point scalars."""
    mesh = pointset()
    mesh.point_data['data'] = mesh.points[:, 2]
    return mesh


def unstructured_with_scalars() -> pv.UnstructuredGrid:
    """Return an unstructured grid carrying point scalars."""
    mesh = unstructured()
    mesh.point_data['data'] = np.linspace(0.0, 1.0, mesh.n_points)
    return mesh


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the widened overloads apply."""
    return True


# Clipping by scalars follows the input class, and `both` hands back both halves
assert_types(poly_with_scalars().clip_scalar(value=0.0), pv.PolyData)
assert_types(poly_with_scalars().clip_scalar(value=0.0, both=True), tuple[pv.PolyData, pv.PolyData])
assert_types(poly_with_scalars().clip_scalar(value=0.0, inplace=True), pv.PolyData)
assert_types(pointset_with_scalars().clip_scalar(value=0.0), pv.PointSet)
assert_types(pointset_with_scalars().clip_scalar(value=0.0, both=True), tuple[pv.PointSet, pv.PointSet])
assert_types(unstructured_with_scalars().clip_scalar(value=0.0), pv.UnstructuredGrid)
assert_types(unstructured_with_scalars().clip_scalar(value=0.0, both=True), tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(image_with_scalars().clip_scalar(value=0.0), pv.UnstructuredGrid)
assert_types(image_with_scalars().clip_scalar(value=0.0, both=True), tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(poly_with_scalars().clip_scalar(value=0.0, both=a_flag()), pv.PolyData | tuple[pv.PolyData, pv.PolyData])
