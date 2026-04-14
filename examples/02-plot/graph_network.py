"""
.. _graph_network_example:

Plot a 3D Graph Network
~~~~~~~~~~~~~~~~~~~~~~~

Render a node-edge network with labels and weighted edges.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

# sphinx_gallery_start_ignore
# point labels are static-only in the docs
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Define nodes and weighted edges
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A small network is enough to show how PyVista can render graph-like data as
# geometry.

labels = ['A', 'B', 'C', 'D', 'E', 'F']
points = np.array(
    [
        (-0.8, 0.0, 0.0),
        (-0.1, 0.7, 0.3),
        (-0.1, -0.7, 0.2),
        (0.8, 0.9, 0.6),
        (0.9, -0.1, 0.8),
        (0.8, -0.9, 0.1),
    ],
)
edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (2, 5), (4, 5)]
weights = np.array([0.9, 0.7, 1.2, 0.8, 1.0, 0.6, 1.1])

segments = np.vstack([points[[start, stop]] for start, stop in edges])
graph = pv.line_segments_from_points(segments)
graph.cell_data['weight'] = weights


# %%
# Render the network
# ~~~~~~~~~~~~~~~~~~
# Edges are colored by weight while the node labels stay anchored in 3D.

pl = pv.Plotter()
pl.add_mesh(
    graph,
    scalars='weight',
    cmap='viridis',
    line_width=8,
    render_lines_as_tubes=True,
)
pl.add_points(points, color='black', point_size=16, render_points_as_spheres=True)
pl.add_point_labels(points, labels, font_size=24, point_size=0, fill_shape=False)
pl.show()
# %%
# .. tags:: plot
