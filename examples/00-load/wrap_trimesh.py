"""
.. _wrap_trimesh_example:

Wrapping Other Objects
~~~~~~~~~~~~~~~~~~~~~~
You can :func:`~pyvista.wrap` several other object types using pyvista including:

- `numpy` arrays
- `trimesh.Trimesh` meshes
- VTK objects

This allows for the "best of both worlds" programming special to
Python due to its modularity.  If there's some limitation of pyvista
(or trimesh), then you can adapt your scripts to use the best features
of more than one module.

"""

# sphinx_gallery_start_ignore
from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Wrap a point cloud composed of random points from numpy
import numpy as np

import pyvista as pv

rng = np.random.default_rng(seed=0)
points = rng.random((30, 3))
cloud = pv.wrap(points)
pv.plot(
    cloud,
    scalars=points[:, 2],
    render_points_as_spheres=True,
    point_size=50,
    opacity=points[:, 0],
    cpos='xz',
)

# %%
# Wrap an instance of Trimesh
import trimesh

points = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
faces = [[0, 1, 2]]
tmesh = trimesh.Trimesh(points, faces=faces, process=False)
mesh = pv.wrap(tmesh)
print(mesh)

# %%
# Wrap an instance of :vtk:`vtkPolyData`

import vtk

points = vtk.vtkPoints()
p = [1.0, 2.0, 3.0]
vertices = vtk.vtkCellArray()
pid = points.InsertNextPoint(p)
vertices.InsertNextCell(1)
vertices.InsertCellPoint(pid)
point = vtk.vtkPolyData()
point.SetPoints(points)
point.SetVerts(vertices)
mesh = pv.wrap(point)
print(mesh)
# %%
# .. tags:: load
