"""One small mesh of every wrappable class, shared by the typing cases."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

import pyvista as pv

_MeshType = TypeVar('_MeshType', bound=pv.DataSet)


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(4, 4, 4))


def rectilinear() -> pv.RectilinearGrid:
    """Return a small rectilinear grid."""
    axis = np.arange(4, dtype=float)
    return pv.RectilinearGrid(axis, axis, axis)


def structured() -> pv.StructuredGrid:
    """Return a small structured grid."""
    axis = np.arange(4, dtype=float)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return pv.StructuredGrid(x, y, z)


def unstructured() -> pv.UnstructuredGrid:
    """Return a small unstructured grid."""
    return image().cast_to_unstructured_grid()


def explicit_structured() -> pv.ExplicitStructuredGrid:
    """Return a small explicit structured grid."""
    grid = structured()
    grid.dimensions = [4, 4, 4]
    return grid.cast_to_explicit_structured_grid()


def pointset() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


def multiblock_poly() -> pv.MultiBlock[pv.PolyData]:
    """Return a composite declared to hold only `PolyData`."""
    return pv.MultiBlock([poly()])


def multiblock_optional_poly() -> pv.MultiBlock[pv.PolyData | None]:
    """Return a composite whose blocks may be missing."""
    return pv.MultiBlock([poly(), None])


def multiblock_image() -> pv.MultiBlock[pv.ImageData]:
    """Return a composite declared to hold only `ImageData`."""
    return pv.MultiBlock([image()])


def multiblock_optional_image() -> pv.MultiBlock[pv.ImageData | None]:
    """Return a composite of grids whose blocks may be missing."""
    return pv.MultiBlock([image(), None])


def multiblock_pointset() -> pv.MultiBlock[pv.PointSet]:
    """Return a composite declared to hold only `PointSet`."""
    return pv.MultiBlock([pointset()])


def multiblock_optional_pointset() -> pv.MultiBlock[pv.PointSet | None]:
    """Return a composite of point clouds whose blocks may be missing."""
    return pv.MultiBlock([pointset(), None])


def multiblock_unstructured() -> pv.MultiBlock[pv.UnstructuredGrid]:
    """Return a composite declared to hold only `UnstructuredGrid`."""
    return pv.MultiBlock([unstructured()])


def multiblock_dataset() -> pv.MultiBlock[pv.DataSet]:
    """Return a composite declared only as holding datasets."""
    return pv.MultiBlock([image()])


def with_arrays(mesh: _MeshType) -> _MeshType:
    """Give a mesh point scalars ``s``, vectors ``v`` and integer ``labels``, keeping its class."""
    mesh.point_data['s'] = mesh.points[:, 0]
    mesh.point_data['v'] = np.tile([1.0, 0.0, 0.0], (mesh.n_points, 1))
    mesh.point_data['labels'] = (np.arange(mesh.n_points) % 3).astype(int)
    mesh.set_active_scalars('s')
    return mesh
