"""
.. _mesh_quality_example:

Computing Mesh Quality
~~~~~~~~~~~~~~~~~~~~~~

Leverage powerful VTK algorithms for computing mesh quality.

Here we will use the :func:`~pyvista.DataSetFilters.compute_cell_quality` filter
to compute the cell qualities. For a full list of the various quality metrics
available, please refer to the documentation for that filter.

Note that many of the measures may only apply to 2D cells or 3D cells. Examples for
meshes with :attr:`~pyvista.CellType.TRIANGLE` and :attr:`~pyvista.CellType.TETRA` cells
are given here.

"""

from __future__ import annotations

from pyvista import examples

# %%
# Triangle Cell Quality
# ---------------------
# Load a :class:`~pyvista.PolyData` mesh and :meth:`~pyvista.PolyDataFilters.decimate`
# it to show coarse :attr:`~pyvista.CellType.TRIANGLE` cells for the example.
# Here we use :meth:`~pyvista.examples.downloads.download_cow`.

mesh = examples.download_cow().triangulate().decimate(0.7)

# %%
# Compute the cell quality. By default, the ``'scaled_jacobian'`` measure is computed.

qual = mesh.compute_cell_quality()
qual

# %%
# Plot the mesh.

plot_kwargs = dict(cmap='bwr', show_edges=True, cpos='xy', zoom=1.3)
qual.plot(scalars='scaled_jacobian', **plot_kwargs)

# %%
# Compute additional measures and plot them all for comparison.

measures = ['area', 'max_angle', 'min_angle', 'shape']
qual = mesh.compute_cell_quality(measures)
for measure in measures:
    qual.plot(scalars=measure, **plot_kwargs)


# %%
# Quality measures like ``'volume'`` do not apply to 2D cells, and a null value
# of ``-1`` is returned.

qual = mesh.compute_cell_quality('volume')
qual.get_data_range('volume')

# %%
# Tetrahedral Cell Quality
# ---------------------
# Load a mesh with :attr:`~pyvista.CellType.TETRA` cells. Here we use
# :meth:`~pyvista.examples.downloads.download_letter_a`.

mesh = examples.download_letter_a()

# %%
# Plot some valid quality measures for tetrahedral cells.

measures = ['volume', 'collapse_ratio', 'jacobian', 'scaled_jacobian']
qual = mesh.compute_cell_quality(measures)
for measure in measures:
    qual.plot(scalars=measure, **plot_kwargs)

# %%
# .. tags:: filter
