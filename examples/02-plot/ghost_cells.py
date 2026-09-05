"""
.. _remove_cells_example:

Remove Cells
~~~~~~~~~~~~

Remove specific cells from a mesh.

Cells can be removed from any dataset by index or with a boolean mask using
:meth:`~pyvista.DataSetFilters.remove_cells`. The output is a
:class:`~pyvista.PolyData` for ``PolyData`` input and an
:class:`~pyvista.UnstructuredGrid` otherwise.

"""

from pyvista import examples

mesh = examples.load_channels()

# %%
# Decide which cells to remove with a criteria (feel free to adjust this
# or manually create this array to remove specific cells).
remove = mesh['facies'] < 1.0

# %%
# Remove the cells and plot the result.
mesh = mesh.remove_cells(remove)
mesh.plot(clim=[0, 4])
# %%
# .. tags:: plot
