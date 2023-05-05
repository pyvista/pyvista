"""
.. _create_dimension_line_example:
Create Dimension Line
~~~~~~~~~~~~~~~~~~~~~

Create a dimension line along 3d structured mesh.

"""

import numpy as np

import pyvista as pv
from pyvista import examples

grid = pv.UnstructuredGrid(examples.hexbeamfile)


plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color='tan')

plotter.enable_parallel_projection()

xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

lines = np.array([[xmax, ymax, zmin], [xmax, ymax, zmax]])

pointa = lines[0:-1]
pointb = lines[1:]
pointc = (pointa + pointb) / 2.0
labels = np.array([str(np.linalg.norm(pointb - pointa))])

plotter.add_point_labels(points=pointc, labels=labels)
plotter.add_lines(lines)

plotter.show()
