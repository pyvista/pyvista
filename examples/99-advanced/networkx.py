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


def create_graph_nodes(graph, positions):
    """Create PyVista spheres for graph nodes.

    Parameters
    ----------
    graph : networkx.Graph
        The NetworkX graph.
    positions : dict
        Node positions from NetworkX layout.

    Returns
    -------
    pyvista.PolyData
        Spheres representing the graph nodes.

    """
    # Convert positions to array
    node_positions = np.array([positions[node] for node in sorted(graph)])

    # Create scalar data (node indices with offset)
    node_scalars = np.array(list(graph.nodes())) + 5

    # Create point cloud and add scalars
    point_cloud = pv.PolyData(node_positions)
    point_cloud['scalars'] = node_scalars

    # Generate spheres for visualization
    return point_cloud.glyph(scale=False, geom=pv.Sphere(radius=0.05))


def create_graph_edges(graph, positions):
    """Create PyVista lines for graph edges.

    Parameters
    ----------
    graph : networkx.Graph
        The NetworkX graph.
    positions : dict
        Node positions from NetworkX layout.

    Returns
    -------
    pyvista.PolyData
        Lines representing the graph edges.

    """
    # Create all edge lines
    edge_lines = []
    for start_node, end_node in graph.edges():
        start_pos = positions[start_node]
        end_pos = positions[end_node]
        line = pv.Line(start_pos, end_pos)
        edge_lines.append(line)

    # Combine all lines into single PolyData object
    if not edge_lines:
        return pv.PolyData()

    combined_lines = edge_lines[0]
    for line in edge_lines[1:]:
        combined_lines += line

    return combined_lines


def visualize_networkx_graph():
    """Create and visualize a NetworkX graph using PyVista."""
    # Create a cycle graph
    cycle_graph = nx.cycle_graph(20)
    graph = nx.convert_node_labels_to_integers(cycle_graph)

    # Generate 3D layout positions
    node_positions = nx.spring_layout(graph, dim=3, seed=1001)

    # Create visualization components
    node_spheres = create_graph_nodes(graph, node_positions)
    edge_lines = create_graph_edges(graph, node_positions)

    # Set up and display the plot
    plotter = pv.Plotter()
    plotter.add_mesh(node_spheres, scalars='scalars', cmap='Blues', show_scalar_bar=True)
    plotter.add_mesh(edge_lines, color='gray', line_width=2)
    plotter.add_axes()
    plotter.show()


if __name__ == '__main__':
    visualize_networkx_graph()
