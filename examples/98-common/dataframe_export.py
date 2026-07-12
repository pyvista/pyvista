"""
.. _dataframe_export_example:

Export Mesh Data to a DataFrame
-------------------------------

Convert a mesh's :attr:`point_data <pyvista.DataSet.point_data>` or
:attr:`cell_data <pyvista.DataSet.cell_data>` to a :class:`pandas.DataFrame`
or :class:`pyarrow.Table` for downstream analytics, export, or interactive
exploration in IDEs like Positron's Data Explorer, JupyterLab, or VS Code's
data viewer.

This example uses a classic CFD-style workflow: a scalar field and a vector
field attached to a mesh, then filtered / aggregated / exported using
pandas idioms.

"""

import numpy as np
from pyvista import examples

# %%
# Load a mesh and attach some data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use the hexbeam fixture mesh and attach a pressure scalar plus a
# velocity vector.

mesh = examples.load_hexbeam()
rng = np.random.default_rng(seed=0)
mesh.point_data['pressure'] = rng.normal(loc=100.0, scale=20.0, size=mesh.n_points)
mesh.point_data['velocity'] = rng.normal(size=(mesh.n_points, 3))

# %%
# Convert to a :class:`pandas.DataFrame`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``mesh.to_pandas()`` returns one row per point. Multi-component arrays are
# expanded to one column per component, named ``{array_name}_{i}``.

df = mesh.to_pandas()
df.head()

# %%
# The DataFrame has ``n_points`` rows and one column per (expanded) array.

df.shape

# %%
# Ad-hoc analytics
# ~~~~~~~~~~~~~~~~
# Since we have a DataFrame, the full pandas API is available: filter,
# derive new columns, summarize.

velocity = df[['velocity_0', 'velocity_1', 'velocity_2']].to_numpy()
df['speed'] = np.linalg.norm(velocity, axis=1)
df[['pressure', 'speed']].describe()

# %%
# Find the 5 highest-pressure points.

df.nlargest(5, 'pressure')[['pressure', 'speed']]

# %%
# Cell-level export
# ~~~~~~~~~~~~~~~~~
# Pass ``association='cell'`` to export :attr:`~pyvista.DataSet.cell_data`
# instead. The result has ``n_cells`` rows.

mesh.cell_data['quality'] = rng.random(mesh.n_cells)
cell_df = mesh.to_pandas('cell')
cell_df.head()

# %%
# Export to disk
# ~~~~~~~~~~~~~~
# A DataFrame gives you one-liner access to every pandas I/O backend:
# Parquet, CSV, Feather, Excel, SQL, HDF5, and more. Commented out here to
# keep the gallery build clean.

# df.to_parquet('point_data.parquet')
# df.to_csv('point_data.csv', index=False)

# %%
# Zero-copy Arrow interchange
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``mesh``, ``mesh.point_data``, ``mesh.cell_data``, and :class:`pyvista.Table`
# all implement the `Arrow PyCapsule interface
# <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_,
# so any Arrow-aware library (polars, DuckDB, ibis, narwhals) can consume them
# without pyvista depending on those libraries. For example, with pyarrow:
import pyarrow as pa

arrow_table = pa.table(mesh)
arrow_table.schema

# %%
# The ``mesh``, ``mesh.point_data``, and ``mesh.cell_data`` objects can also be
# opened directly in data-science IDE variable explorers (Positron, Jupyter,
# VS Code) after calling :meth:`~pyvista.DataSet.to_pandas`. The returned
# DataFrame renders as an interactive, sortable, filterable table.
