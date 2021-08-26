"""
Plot Open Street Map Data
~~~~~~~~~~~~~~~~~~~~~~~~~

This was originally posted to `pyvista/pyvista-support#486 <https://github.com/pyvista/pyvista-support/issues/486>`_.

Be sure to check out `osmnx <https://github.com/gboeing/osmnx>`_

Start by generating a graph from an address.

"""

import osmnx as ox
import numpy as np
import pyvista as pv

ox.config(use_cache=False)
address = 'Holzgerlingen DE'
graph = ox.graph_from_address(address, dist=500, network_type='drive')

###############################################################################
# Next, convert the edges into pyvista lines.

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
combined_lines.plot(line_width=5, cpos='xy', background='w', color='k')

