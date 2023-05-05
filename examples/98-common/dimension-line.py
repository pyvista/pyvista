"""
.. _dimension_line_example:
Dimension Line
~~~~~~~~~~~~~~

Create a dimension line along 3d structured mesh.

"""

import pyvista as pv
from pyvista import examples

grid = pv.UnstructuredGrid(examples.hexbeamfile)


plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color='tan')

plotter.enable_parallel_projection()

xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

plotter.add_dimension_lines([xmax, ymax, zmin], [xmax, ymax, zmax])
plotter.add_dimension_lines([xmin, ymin, zmin], [xmin, ymin, zmax])

plotter.show()
