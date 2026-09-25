"""Typing cases for :meth:`pyvista.Plotter.add_mesh`."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import trimesh
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from pyvista import examples


def as_vtk_mesh() -> _vtk.vtkPolyData:
    """Return a mesh typed only as the VTK class."""
    return pv.Sphere()


def a_trimesh() -> trimesh.Trimesh:
    """Return a single-triangle trimesh mesh."""
    return trimesh.Trimesh([[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]], process=False)


def a_meshio_mesh() -> meshio.Mesh:
    """Return a single-triangle meshio mesh."""
    return meshio.Mesh(points=np.eye(3), cells=[('triangle', np.array([[0, 1, 2]]))])


assert_types(pv.Plotter().add_mesh(pv.Sphere()), pv.Actor)
assert_types(pv.Plotter().add_mesh(pv.MultiBlock([pv.Sphere()])), pv.Actor)
assert_types(pv.Plotter().add_mesh(as_vtk_mesh()), pv.Actor)
assert_types(pv.Plotter().add_mesh(a_trimesh()), pv.Actor)
assert_types(pv.Plotter().add_mesh(a_meshio_mesh()), pv.Actor)
assert_types(pv.Plotter().add_mesh(np.random.default_rng().random((10, 3))), pv.Actor)
assert_types(pv.Plotter().add_mesh(examples.antfile), pv.Actor)
assert_types(pv.Plotter().add_mesh(Path(examples.antfile)), pv.Actor)
