"""
.. _distance_measurement_example:

Measuring distance
~~~~~~~~~~~~~~~~~~
This example demonstrates how to measure distance between two points.
:func:`add_distance_widget() <pyvista.Plotter.add_distance_widget>`.

"""

import pyvista as pv


cube = pv.Cube()
cube2 = pv.Cube([10, 10, 0])

sphere = pv.Sphere()
sphere.translate([10, 0, 0])

p = pv.Plotter()
p.add_mesh(cube)
p.add_mesh(cube2)
p.add_mesh(sphere)
func = lambda start, end, distance: distance
p.add_distance_widget(callback=func)
p.show(auto_close=False)

