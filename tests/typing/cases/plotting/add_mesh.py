"""Typing cases for :meth:`pyvista.Plotter.add_mesh`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import meshio
import numpy as np
import trimesh
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from pyvista import examples

if TYPE_CHECKING:
    from numpy.typing import NDArray


def as_vtk_mesh() -> _vtk.vtkPolyData:
    """Return a mesh typed only as the VTK class."""
    return pv.Sphere()


def a_trimesh() -> trimesh.Trimesh:
    """Return a single-triangle trimesh mesh."""
    return trimesh.Trimesh([[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]], process=False)


def a_meshio_mesh() -> meshio.Mesh:
    """Return a single-triangle meshio mesh."""
    return meshio.Mesh(points=np.eye(3), cells=[('triangle', np.array([[0, 1, 2]]))])


def a_mask() -> NDArray[np.bool_]:
    """Return a boolean mask volume."""
    return np.zeros((2, 2, 2), dtype=np.bool_)


def a_label_volume() -> NDArray[np.uint8]:
    """Return a label volume."""
    return np.zeros((2, 2, 2), dtype=np.uint8)


def int64_scalars() -> NDArray[np.int64]:
    """Return int64 point scalars for a sphere."""
    return np.arange(pv.Sphere().n_points, dtype=np.int64)


def uint8_rgb_scalars() -> NDArray[np.uint8]:
    """Return uint8 RGB point scalars for a sphere."""
    return np.zeros((pv.Sphere().n_points, 3), dtype=np.uint8)


def bool_scalars() -> NDArray[np.bool_]:
    """Return boolean point scalars for a sphere."""
    return np.zeros(pv.Sphere().n_points, dtype=np.bool_)


def uint8_texture() -> NDArray[np.uint8]:
    """Return a uint8 RGB texture image."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


assert_types(pv.Plotter().add_mesh(pv.Sphere()), pv.Actor)
assert_types(pv.Plotter().add_mesh(pv.MultiBlock([pv.Sphere()])), pv.Actor)
assert_types(pv.Plotter().add_mesh(as_vtk_mesh()), pv.Actor)
assert_types(pv.Plotter().add_mesh(a_trimesh()), pv.Actor)
assert_types(pv.Plotter().add_mesh(a_meshio_mesh()), pv.Actor)
assert_types(pv.Plotter().add_mesh(np.random.default_rng().random((10, 3))), pv.Actor)
assert_types(pv.Plotter().add_mesh([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), pv.Actor)
assert_types(pv.Plotter().add_mesh(examples.antfile), pv.Actor)
assert_types(pv.Plotter().add_mesh(Path(examples.antfile)), pv.Actor)
assert_types(pv.Plotter().add_mesh(a_mask()), pv.Actor)
assert_types(pv.Plotter().add_mesh(a_label_volume()), pv.Actor)
assert_types(pv.Plotter().add_mesh(pv.Sphere(), scalars=int64_scalars()), pv.Actor)
assert_types(pv.Plotter().add_mesh(pv.Sphere(), scalars=uint8_rgb_scalars(), rgb=True), pv.Actor)
assert_types(pv.Plotter().add_mesh(pv.Sphere(), scalars=bool_scalars()), pv.Actor)
assert_types(pv.Plotter().add_mesh(pv.Plane(), texture=uint8_texture()), pv.Actor)
