"""
.. _clip_with_plane_box_example:

Clipping with Planes & Boxes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clip/cut any dataset using planes or boxes.
"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import pyvista as pv
from pyvista import examples

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Clip with Plane
# +++++++++++++++
#
# Clip any dataset by a user defined plane using the
# :func:`pyvista.DataObjectFilters.clip` filter
dataset = examples.download_bunny_coarse()
clipped = dataset.clip('y', invert=False)

pl = pv.Plotter()
pl.add_mesh(dataset, style='wireframe', color='blue', label='Input')
pl.add_mesh(clipped, label='Clipped')
pl.add_legend()
pl.camera_position = pv.CameraPosition(
    position=(0.24, 0.32, 0.7),
    focal_point=(0.02, 0.03, -0.02),
    viewup=(-0.12, 0.93, -0.34),
)
pl.show()


# %%
# Clip with Bounds
# ++++++++++++++++
#
# Clip any dataset by a set of XYZ bounds using the
# :func:`pyvista.DataObjectFilters.clip_box` filter.
dataset = examples.download_office()

bounds = [2, 4.5, 2, 4.5, 1, 3]
clipped = dataset.clip_box(bounds)

pl = pv.Plotter()
pl.add_mesh(dataset, style='wireframe', color='blue', label='Input')
pl.add_mesh(clipped, label='Clipped')
pl.add_legend()
pl.show()


# %%
# Clip with Rotated Box
# +++++++++++++++++++++
#
# Clip any dataset by an arbitrarily rotated solid box using the
# :func:`pyvista.DataObjectFilters.clip_box` filter.
mesh = examples.load_airplane()

# Use `pv.Box()` or `pv.Cube()` to create a region of interest
roi = pv.Cube(
    center=(0.9e3, 0.2e3, mesh.center[2]), x_length=500, y_length=500, z_length=500
)
roi.rotate_z(33, inplace=True)

pl = pv.Plotter()
pl.add_mesh(roi, opacity=0.75, color='red')
pl.add_mesh(mesh, opacity=0.5)
pl.show()

# %%
# Run the box clipping algorithm
extracted = mesh.clip_box(roi, invert=False)

pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(roi, opacity=0.75, color='red')
pl.add_mesh(mesh)
pl.subplot(0, 1)
pl.add_mesh(extracted)
pl.add_mesh(roi, opacity=0.75, color='red')
pl.link_views()
pl.view_isometric()
pl.show()

# %%
# Crinkled Clipping
# +++++++++++++++++
# Crinkled clipping is useful if you don't want the clip filter to truly clip
# cells on the boundary, but want to preserve the input cell structure and to
# pass the entire cell on through the boundary.
#
# This option is available for :func:`pyvista.DataObjectFilters.clip`,
# :func:`pyvista.DataObjectFilters.clip_box`, and
# :func:`pyvista.DataSetFilters.clip_surface`, but not available when clipping
# by scalar in :func:`pyvista.DataSetFilters.clip_scalar`.

# Input mesh
mesh = pv.Wavelet()

# %%
# Define clipping plane
normal = (1, 1, 1)
plane = pv.Plane(i_size=30, j_size=30, direction=normal)

# %%
# Perform a standard clip
clipped = mesh.clip(normal=normal)

# %%
# Perform a crinkled clip
crinkled = mesh.clip(normal=normal, crinkle=True)

# %%
# Plot comparison
pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(clipped, show_edges=True)
pl.add_mesh(plane.extract_feature_edges(), color='r')
pl.subplot(0, 1)
pl.add_mesh(crinkled, show_edges=True)
pl.add_mesh(plane.extract_feature_edges(), color='r')
pl.link_views()
pl.show()
# %%
# .. tags:: filter
