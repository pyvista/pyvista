"""
.. _orientation_rotate_widget_example:

Orientation Rotate Widget
~~~~~~~~~~~~~~~~~~~~~~~~~

Use an orientation rotate widget.

"""
import pyvista as pv

mesh = pv.Cube(x_length=0.5, y_length=0.5, z_length=0.5)
p = pv.Plotter()
p.add_mesh(mesh)
p.add_rotate_orientation_widget()
p.show()
