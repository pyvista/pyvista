"""
.. _open_street_map_example:

Plot Open Street Map Data
~~~~~~~~~~~~~~~~~~~~~~~~~

This was originally posted to `pyvista/pyvista-support#486 <https://github.com/pyvista/pyvista-support/issues/486>`_.

Be sure to check out `osmnx <https://github.com/gboeing/osmnx>`_

Start by generating a graph from an address.

"""

import osmnx as ox
import numpy as np

import pyvista as pv

###############################################################################
# Read in the graph directly from the Open Street Map server.

# address = 'Holzgerlingen DE'
# graph = ox.graph_from_address(address, dist=500, network_type='drive')
# pickle.dump(graph, open('/tmp/tmp.p', 'wb'))

# Alternatively, use the pickeled graph included in our examples.
from pyvista import examples
graph = examples.download_osmnx_graph()


###############################################################################
# Next, convert the edges into pyvista lines using
# :func:`pyvista.lines_from_points`.

nodes, edges = ox.graph_to_gdfs(graph)
lines = []

# convert each edge into a line
for idx, row in edges.iterrows():
    x_pts = row['geometry'].xy[0]
    y_pts = row['geometry'].xy[1]
    z_pts = np.zeros(len(x_pts))
    pts = np.column_stack((x_pts, y_pts, z_pts))
    line = pv.lines_from_points(pts)
    lines.append(line)


###############################################################################
# Finally, merge the lines and plot

combined_lines = lines[0].merge(lines[1:])
combined_lines.plot(line_width=3, cpos='xy')
