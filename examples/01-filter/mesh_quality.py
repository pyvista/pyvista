"""
.. _mesh_quality_example:

Computing Mesh Quality
~~~~~~~~~~~~~~~~~~~~~~

Leverage powerful VTK algorithms for computing mesh quality.

Here we will use the :func:`~pyvista.DataSetFilters.compute_cell_quality` filter
to compute the cell qualities. For a full list of the various quality metrics
available, please refer to the documentation for that filter.
"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

mesh = examples.download_cow().triangulate().decimate(0.7)

# %%
# Compute the cell quality. By default, the ``'scaled_jacobian'`` measure is computed.

mesh = mesh.compute_cell_quality()
mesh

# %%
# Plot the mesh. We define a custom method to add a mesh and set default plot values.


def add_mesh(plotter, mesh_, scalars=None, cmap='bwr', show_edges=True, zoom=1.3):
    plotter.add_mesh(mesh_, scalars=scalars, cmap=cmap, show_edges=show_edges)
    plotter.view_xy()
    plotter.camera.zoom(zoom)


pl = pv.Plotter()
add_mesh(pl, mesh, scalars='scaled_jacobian')
pl.show()

# %%
# Note that there are many different quality measures, many of which may only apply to
# 2D cells or 3D cells (see :class:`~pyvista.CellType` for more information about cell
# types). For :attr:`~pyvista.CellType.TRIANGLE` and :attr:`~pyvista.CellType.QUAD`
# cells, the following measures produce meaningful values.

measures = ['area', 'max_angle', 'min_angle', 'shape']
mesh = mesh.compute_cell_quality(measures)
mesh


# %%
pl = pv.Plotter(shape=(2, 2))
pl.link_views()
pl.subplot(0, 0)
add_mesh(pl, mesh, scalars=measures[0])
pl.subplot(0, 1)
add_mesh(pl, mesh, scalars=measures[1])
pl.subplot(1, 0)
add_mesh(pl, mesh, scalars=measures[2])
pl.subplot(1, 1)
add_mesh(pl, mesh, scalars=measures[3])
pl.show()
