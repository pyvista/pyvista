"""
.. _orientation_rotate_widget_example:

Orientation Rotate Widget
~~~~~~~~~~~~~~~~~~~~~~~~~

Use an orientation rotate widget.

"""

import vtk

import pyvista as pv

mesh = pv.Cube(x_length=0.5, y_length=0.5, z_length=0.5)
p = pv.Plotter()
p.add_mesh(mesh)

orientation_rep = vtk.vtkOrientationRepresentation()

orientation_widget = vtk.vtkOrientationWidget()
orientation_widget.SetInteractor(p.iren.interactor)
orientation_widget.SetCurrentRenderer(p.renderer)
orientation_widget.SetRepresentation(orientation_rep)
orientation_widget.On()

p.show()
