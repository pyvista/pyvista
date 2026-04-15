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
from pyvista import examples

# sphinx_gallery_thumbnail_number = 2

# %%
# Load a reference surface
# ~~~~~~~~~~~~~~~~~~~~~~~~
# The :func:`~pyvista.examples.downloads.download_action_figure` scan is an
# asymmetric reference mesh.

target = examples.download_action_figure()
target


# %%
# Transform a copy away from the reference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The transformed copy stands in for an incoming scan that needs to be
# registered back onto the reference.

offset = np.array(target.length) * 0.4
transform = pv.Transform().rotate_x(25).rotate_z(-35).translate((offset, -offset, offset))
source = target.transform(transform, inplace=False)

pl = pv.Plotter()
pl.add_mesh(target, style='wireframe', color='black', line_width=3)
pl.add_mesh(source, color='tomato', opacity=0.8)
pl.show()


# %%
# Recover the rigid transform
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :func:`pyvista.DataSetFilters.align` runs ICP and returns both the aligned
# mesh and the recovered transform matrix.

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
# The returned matrix maps the transformed copy back onto the reference.

np.round(matrix, 3)


# %%
# Measure the residual distances
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A successful registration leaves a small point-to-surface residual.

aligned['distance_to_target'].mean(), aligned['distance_to_target'].max()
# %%
# .. tags:: filter
