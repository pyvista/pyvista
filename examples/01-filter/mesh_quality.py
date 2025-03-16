"""
.. _mesh_quality_example:

Computing Mesh Quality
~~~~~~~~~~~~~~~~~~~~~~

Leverage powerful VTK algorithms for computing mesh quality.

Here we will use the :func:`~pyvista.DataObjectFilters.compute_cell_quality` filter
to compute the cell qualities. For a full list of the various quality metrics
available, please refer to the documentation for that filter.

Note that many of the measures may only apply to 2D cells or 3D cells. Examples for
meshes with :attr:`~pyvista.CellType.TRIANGLE` and :attr:`~pyvista.CellType.TETRA` cells
are given here.

"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# %%
# Triangle Cell Quality
# ---------------------
# Load a :class:`~pyvista.PolyData` mesh and :meth:`~pyvista.PolyDataFilters.decimate`
# it to show coarse :attr:`~pyvista.CellType.TRIANGLE` cells for the example.
# Here we use :meth:`~pyvista.examples.downloads.download_cow`.

mesh = examples.download_cow().triangulate().decimate(0.7)

# %%
# Compute some valid measures for triangle cells.

measures = ['area', 'shape', 'min_angle', 'max_angle']
qual = mesh.compute_cell_quality(measures)

# %%
# Plot the meshes in subplots for comparison. We define a custom method
# for adding each mesh to each subplot.


def add_mesh(plotter, mesh, scalars=None, cmap='bwr', show_edges=True):
    # Create a copy to avoid reusing the same mesh in different plots
    copied = mesh.copy(deep=False)
    plotter.add_mesh(copied, scalars=scalars, cmap=cmap, show_edges=show_edges)
    plotter.view_xy()


pl = pv.Plotter(shape=(2, 2))
pl.link_views()
pl.subplot(0, 0)
add_mesh(pl, qual, scalars=measures[0])
pl.subplot(0, 1)
add_mesh(pl, qual, scalars=measures[1])
pl.subplot(1, 0)
add_mesh(pl, qual, scalars=measures[2])
pl.subplot(1, 1)
add_mesh(pl, qual, scalars=measures[3])
pl.show()


# %%
# Quality measures like ``'volume'`` do not apply to 2D cells, and a null value
# of ``-1`` is returned.

qual = mesh.compute_cell_quality(['volume'])
qual.get_data_range('volume')

# %%
# Tetrahedron Cell Quality
# ------------------------
# Load a mesh with :attr:`~pyvista.CellType.TETRA` cells. Here we use
# :meth:`~pyvista.examples.downloads.download_letter_a`.

mesh = examples.download_letter_a()

# %%
# Plot some valid quality measures for tetrahedral cells.

measures = ['volume', 'collapse_ratio', 'jacobian', 'scaled_jacobian']
qual = mesh.compute_cell_quality(measures)

pl = pv.Plotter(shape=(2, 2))
pl.link_views()
pl.subplot(0, 0)
add_mesh(pl, qual, scalars=measures[0])
pl.subplot(0, 1)
add_mesh(pl, qual, scalars=measures[1])
pl.subplot(1, 0)
add_mesh(pl, qual, scalars=measures[2])
pl.subplot(1, 1)
add_mesh(pl, qual, scalars=measures[3])
pl.show()

# %%
# .. tags:: filter
