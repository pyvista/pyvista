"""
.. _floors_example:

Plot with Floors
~~~~~~~~~~~~~~~~

Add a floor/wall at the boundary of the rendering scene.
"""

import pyvista as pv
from pyvista import examples

mesh = examples.download_dragon()

plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.add_floor('-y')
plotter.add_floor('-z')
plotter.show()
