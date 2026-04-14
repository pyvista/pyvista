"""
.. _icp_registration_example:

Register a Surface with ICP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recover the rigid transform between two surfaces with the iterative closest
point implementation behind :func:`pyvista.DataSetFilters.align`.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# sphinx_gallery_thumbnail_number = 2

# %%
# Create a reference surface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Start with a simple asymmetric surface so the registration problem is
# self-contained and does not depend on any downloaded assets.

target = pv.Text3D('PV', depth=0.3).triangulate()
target


# %%
# Transform a copy away from the reference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The transformed copy plays the role of the incoming scan that we would like to
# register back onto the reference surface.

transform = pv.Transform().rotate_x(25).rotate_z(-35).translate((1.5, -0.3, 0.7))
source = target.transform(transform, inplace=False)

pl = pv.Plotter()
pl.add_mesh(target, style='wireframe', color='black', line_width=3)
pl.add_mesh(source, color='tomato', opacity=0.8)
pl.show()


# %%
# Recover the rigid transform
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use :func:`pyvista.DataSetFilters.align` to register the transformed copy back
# to the reference mesh.

aligned, matrix = source.align(target, return_matrix=True)

_, closest_points = target.find_closest_cell(aligned.points, return_closest_point=True)
aligned['distance_to_target'] = np.linalg.norm(aligned.points - closest_points, axis=1)

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_mesh(target, style='wireframe', color='black', line_width=3)
pl.add_mesh(source, color='tomato', opacity=0.8)
pl.subplot(0, 1)
pl.add_mesh(target, style='wireframe', color='black', line_width=3)
pl.add_mesh(aligned, scalars='distance_to_target', cmap='viridis')
pl.link_views()
pl.show()


# %%
# Inspect the recovered transform
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The returned matrix maps the transformed surface back toward the reference
# surface.

np.round(matrix, 3)


# %%
# Measure the residual distances
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A successful registration leaves only a small point-to-surface residual.

aligned['distance_to_target'].mean(), aligned['distance_to_target'].max()
# %%
# .. tags:: filter
