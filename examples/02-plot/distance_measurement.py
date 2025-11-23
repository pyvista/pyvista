"""
.. _distance_measurement_example:

Measuring distance
~~~~~~~~~~~~~~~~~~
This example demonstrates how to measure distance between two points.
:func:`add_measurement_widget() <pyvista.Plotter.add_measurement_widget>`.

"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

import pyvista as pv

cube = pv.Cube()
cube2 = pv.Cube(center=[10, 10, 0])

pl = pv.Plotter()
pl.add_mesh(cube)
pl.add_mesh(cube2)


def callback(a, b, distance):  # noqa: ARG001
    pl.add_text(f'Distance: {distance:.2f}', name='dist')


pl.add_measurement_widget(callback)
pl.show()
# %%
# .. tags:: plot
