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

p = pv.Plotter(notebook=0)
p.add_mesh(cube)
p.add_mesh(cube2)


def callback(a, b, distance):
    p.add_text(f'Distance: {distance:.2f}', name='dist')


p.add_distance_widget(callback)
p.show()
