"""
.. _contouring_example:

Contouring
~~~~~~~~~~

Generate iso-lines or -surfaces for the scalars of a surface or volume.

3D meshes can have 2D iso-surfaces of a scalar field extracted and 2D surface
meshes can have 1D iso-lines of a scalar field extracted.
"""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples

# %%
# Iso-Lines
# +++++++++
#
# Let's extract 1D iso-lines of a scalar field from a 2D surface mesh.
mesh = examples.load_random_hills()

contours = mesh.contour()

pl = pv.Plotter()
pl.add_mesh(mesh, opacity=0.85)
pl.add_mesh(contours, color='white', line_width=5)
pl.show()


# %%
# Iso-Surfaces
# ++++++++++++
#
# Let's extract 2D iso-surfaces of a scalar field from a 3D mesh.
mesh = examples.download_embryo()

contours = mesh.contour(np.linspace(50, 200, 5))

pl = pv.Plotter()
pl.add_mesh(mesh.outline(), color='k')
pl.add_mesh(contours, opacity=0.25, clim=[0, 200])
pl.camera_position = pv.CameraPosition(
    position=(-130.99381142132086, 644.4868354828589, 163.80447435848686),
    focal_point=(125.21748748157661, 123.94368717158413, 108.83283586619626),
    viewup=(0.2780372840777734, 0.03547871361794171, 0.9599148553609699),
)
pl.show()


# %%
# Banded Contours
# +++++++++++++++
# Create banded contours for surface meshes using :func:`~pyvista.PolyDataFilters.contour_banded`.
mesh = examples.load_random_hills()

# %%
# Set number of contours and produce mesh and lines
n_contours = 8
contours, edges = mesh.contour_banded(n_contours)

# %%
# Also make normal vectors
arrows = mesh.glyph(scale='Normals', orient='Normals', tolerance=0.05)

# %%

# Common display arguments
dargs = dict(scalars='Elevation', n_colors=n_contours - 1, cmap='Set3')

pl = pv.Plotter()
pl.add_mesh(edges, line_width=5, render_lines_as_tubes=True, color='k')
pl.add_mesh(contours, **dargs)
pl.add_mesh(arrows, **dargs)
pl.show()

# %%
# Contours from a label map
# +++++++++++++++++++++++++
#
# Create labeled surfaces from 3D label maps (e.f. multi-label image segmentation)
# using :func:`~pyvista.ImageDataFilters.contour_labels`.
# Requires VTK version 9.3
if pv.vtk_version_info >= (9, 3):
    label_map = pv.examples.load_frog_tissues()
    mesh = label_map.contour_labels()
    mesh.plot(cmap='glasbey', cpos='yx', categories=True)
# %%
# .. tags:: filter
