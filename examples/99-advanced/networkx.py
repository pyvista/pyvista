"""
.. _networkx_example:

NetworkX Graph Visualization
----------------------------

This example demonstrates how to visualize a NetworkX graph using PyVista.
It creates a 3D representation of a cycle graph with nodes represented as spheres
and edges as lines. The nodes are colored based on their indices with an offset.

"""

from __future__ import annotations

import networkx as nx
import numpy as np

import pyvista as pv

# Create a graph
H = nx.cycle_graph(20)
G = nx.convert_node_labels_to_integers(H)
pos = nx.spring_layout(G, dim=3, seed=1001)
xyz = np.array([pos[v] for v in sorted(G)])

# Scalar data (node indices with offset)
scalars = np.array(list(G.nodes())) + 5

# Treat the node positions as PyVista point data
point_cloud = pv.PolyData(xyz)
point_cloud['scalars'] = scalars

# Visualize nodes using spheres
spheres = point_cloud.glyph(scale=False, geom=pv.Sphere(radius=0.05))

# Create lines for edges
edges = []
for i, j in G.edges():
    edge = np.array([pos[i], pos[j]])
    edges.append(edge)

# Combine all edges into a single PolyData object
lines = []
for edge in edges:
    line = pv.Line(edge[0], edge[1])
    lines.append(line)
graph_lines = lines[0]
for l in lines[1:]:
    graph_lines += l

# Plot the graph
plotter = pv.Plotter()
plotter.add_mesh(spheres, scalars='scalars', cmap='Blues', show_scalar_bar=True)
plotter.add_mesh(graph_lines, color='gray', line_width=2)
plotter.add_axes()
plotter.show()
