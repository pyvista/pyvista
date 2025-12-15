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
# Wrap Point Arrays
# -----------------
# Wrap a point cloud composed of random points from numpy.
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
# Wrap Trimesh Objects
# --------------------
# Create a simple :class:`trimesh.Trimesh` and wrap it.

import trimesh

points = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
faces = [[0, 1, 2]]
tmesh = trimesh.Trimesh(points, faces=faces, process=False)
mesh = pv.wrap(tmesh)
print(mesh)

# %%
# We can also convert to a :class:`trimesh.Trimesh` mesh with :func:`~pyvista.to_trimesh`.
# First create :class:`~pyvista.PolyData` with data.
mesh = pv.Sphere()
mesh.point_data['point_ids'] = np.arange(mesh.n_points)
mesh.cell_data['cell_ids'] = np.arange(mesh.n_cells)
mesh.field_data['misc'] = np.array([1, 2, 3])
mesh.user_dict['name'] = 'sphere'
mesh.user_dict['number'] = 42

# %%
# Now convert it.
tmesh = pv.to_trimesh(mesh)
print(tmesh)

# %%
# Point data is accessible via ``vertex_attributes``.
print(tmesh.vertex_attributes.keys())

# %%
# Cell data is accessible via ``face_attributes``.
print(tmesh.face_attributes.keys())

# %%
# Field data is accessible via ``metadata``. Both field data keys and
# :attr:`~pyvista.DataObject.user_dict` keys are stored.
print(tmesh.metadata)

# %%
# Use :func:`~pyvista.from_trimesh` to convert it back to a :class:`~pyvista.PolyData` mesh.
# This is the same as using :func:`~pyvista.wrap`.
pvmesh = pv.from_trimesh(tmesh)

# The trimesh data is recovered as point data, cell data, and field data, including field data
# stored in the :attr:`~pyvista.DataObject.user_dict`.
print(pvmesh.point_data.keys())
print(pvmesh.cell_data.keys())
print(pvmesh.field_data.keys())
print(pvmesh.user_dict)

# %%
# Wrap VTK Meshes
# ---------------
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
