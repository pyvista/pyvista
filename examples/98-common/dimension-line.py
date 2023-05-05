"""
.. _dimension_line_example:
Dimension Line
~~~~~~~~~~~~~~

Create a dimension line along 2d structured mesh.

"""

import numpy as np

import pyvista as pv
from pyvista import examples

grid = pv.UnstructuredGrid(examples.hexbeamfile)


plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color='tan')

plotter.enable_parallel_projection()
plotter.show()
