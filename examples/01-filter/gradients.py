"""
.. _gradients_example:

Compute Gradients of a Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimate the gradient of a scalar or vector field in a data set.

The ordering for the output gradient tuple will be
{du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz} for
an input array {u, v, w}.

Showing the :func:`pyvista.DataSetFilters.compute_derivative` filter.
"""

from __future__ import annotations

import numpy as np

# sphinx_gallery_thumbnail_number = 1
import pyvista as pv
from pyvista import examples

# A vtkStructuredGrid - but could be any mesh type
mesh = examples.download_carotid()
mesh
# %%
# Now compute the gradients of the ``vectors`` vector field in the point data
# of that mesh. This is as simple as calling
# :func:`pyvista.DataSetFilters.compute_derivative`.
mesh_g = mesh.compute_derivative(scalars='vectors')
mesh_g['gradient']

# %%
# .. note:: You can also use :func:`pyvista.DataSetFilters.compute_derivative` for
#   computing other derivative based quantities, such as divergence, vorticity,
#   and Q-criterion. See function documentation for options.

# %%
# ``mesh_g["gradient"]`` is an ``N`` by 9 NumPy array of the gradients, so we
# could make a dictionary of NumPy arrays of the gradients like:


def gradients_to_dict(arr):
    """Label the gradients into a dictionary."""
    keys = np.array(
        ['du/dx', 'du/dy', 'du/dz', 'dv/dx', 'dv/dy', 'dv/dz', 'dw/dx', 'dw/dy', 'dw/dz'],
    )
    keys = keys.reshape((3, 3))[:, : arr.shape[1]].ravel()
    return dict(zip(keys, mesh_g['gradient'].T))


gradients = gradients_to_dict(mesh_g['gradient'])
gradients

# %%
# And we can add all of those components as individual arrays back to the mesh
# by:
mesh_g.point_data.update(gradients)
mesh_g

# %%

keys = np.array(list(gradients.keys())).reshape(3, 3)

p = pv.Plotter(shape=keys.shape)
for (i, j), name in np.ndenumerate(keys):
    p.subplot(i, j)
    p.add_mesh(mesh_g.contour(scalars=name), scalars=name, opacity=0.75)
    p.add_mesh(mesh_g.outline(), color='k')
p.link_views()
p.view_isometric()
p.show()


# %%
# And there you have it, the gradients for a vector field. We could also do
# this for a scalar  field like for the ``scalars`` field in the given dataset.
mesh_g = mesh.compute_derivative(scalars='scalars')

gradients = gradients_to_dict(mesh_g['gradient'])
gradients

# %%

mesh_g.point_data.update(gradients)

keys = np.array(list(gradients.keys())).reshape(1, 3)

p = pv.Plotter(shape=keys.shape)

for (i, j), name in np.ndenumerate(keys):
    p.subplot(i, j)
    p.add_mesh(mesh_g.contour(scalars=name), scalars=name, opacity=0.75)
    p.add_mesh(mesh_g.outline(), color='k')
p.link_views()
p.view_isometric()
p.show()
# %%
# .. tags:: filter
