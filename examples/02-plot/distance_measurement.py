"""
.. _distance_measurement_example:

Measuring distance
~~~~~~~~~~~~~~~~~~

Measure distance between two points using :func:`~pyvista.Plotter.add_measurement_widget`.

"""

import pyvista as pv

cube = pv.Cube()
cube2 = pv.Cube(center=[10, 10, 0])

pl = pv.Plotter()
pl.add_mesh(cube)
pl.add_mesh(cube2)


def callback(a, b, distance):
    pl.add_text(f'Distance: {distance:.2f}', name='dist')


pl.add_measurement_widget(callback)
pl.show()
# %%
# .. tags:: plot
