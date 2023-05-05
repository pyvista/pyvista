"""
.. _create_dimension_line_example:
Create Dimension Line
~~~~~~~~~~~~~~~~~~~~~

Create a 2d dimension line along 2d structured mesh.

"""

import numpy as np

import pyvista as pv

pv.set_plot_theme("document")

xrng = np.arange(-10, 10, 2)
yrng = np.arange(-10, 10, 5)
grid = pv.RectilinearGrid(xrng, yrng)

plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color='tan')

plotter.enable_parallel_projection()

lines = np.array([[xrng[0], yrng[-1], 0.0], [xrng[-1], yrng[-1], 0.0]])
lines += np.array([[0.0, 4.0, 0.0], [0.0, 4.0, 0.0]])
mlines = pv.MultipleLines(lines)
mlines["Normal"] = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
arrows = mlines.glyph(geom=pv.Line(), scale="Normal", factor=2.0, orient="Normal")

pointa = lines[0:-1]
pointb = lines[1:]
pointc = (pointa + pointb) / 2.0
labels = np.array([str(np.linalg.norm(pointb - pointa))])

plotter.add_point_labels(points=pointc, labels=labels, shape_color="white")
plotter.add_lines(lines, color="black", width=2)
plotter.add_mesh(arrows, color="black")
plotter.show(cpos="xy")
