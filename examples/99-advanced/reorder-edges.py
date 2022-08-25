"""
.. _reorder_edges_example:

Reorder Edges
~~~~~~~~~~~~~

This example demonstrates how to reorder edges based on connectivity.

Boundary edges are extracted from a :class:`pyvista.PolyData` are unordered. We
can use a simple mapping array based on :attr:`lines <pyvista.PolyData.lines>`
to reorder the edges to make them sequential.

"""

import numpy as np

import pyvista as pv
from pyvista import examples

###############################################################################
# Clip dataset
# ~~~~~~~~~~~~
# Download an example dataset and clip it so we have open edges using the
# :func:`clip <pyvista.DataSet.clip>` filter.
dataset = examples.download_bunny_coarse()
mesh = dataset.clip('y', invert=True)
mesh.plot()


###############################################################################
# Extract and reorder edges
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract the edges and show they are unordered.
#

edge_data = mesh.extract_feature_edges(
    boundary_edges=True, feature_edges=False, manifold_edges=False
)

# show the edges are unordered
pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_mesh(
    edge_data,
    scalars=range(edge_data.n_points),
    line_width=10,
    show_scalar_bar=False,
    cmap='plasma',
)
pl.show()


###############################################################################
# Reorder edges
# ~~~~~~~~~~~~~
# Reorder the edges and plot it.

edges = edge_data.lines.reshape(-1, 3)[:, 1:]  # n_edges x 2
u, v = edges.T

# create a simple adjcency mapper array
adj = np.empty(u.shape, dtype=u.dtype)
adj[u] = v

# Walk through each adjacent edge.
v = edges[0, 0]  # Starting vertex
vert_idxs = [v]
for _ in range(len(edges)):
    v = adj[v]
    vert_idxs.append(v)

# create a new edges polydata
new_lines = np.empty((edges.shape[0], 3), dtype=edges.dtype)
new_lines[:, 0] = 2
new_lines[:, 1] = range(edges.shape[0])
new_lines[:, 2] = range(1, edges.shape[0] + 1)
new_edges = pv.PolyData(edge_data.points[vert_idxs], lines=new_lines)

# plot just the edges
new_edges.plot(
    scalars=range(new_edges.n_points), line_width=10, show_scalar_bar=False, cmap='plasma'
)


# ###############################################################################
# Plot it with the original mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_mesh(
    new_edges,
    scalars=range(new_edges.n_points),
    line_width=10,
    show_scalar_bar=False,
    cmap='plasma',
)
pl.show()
