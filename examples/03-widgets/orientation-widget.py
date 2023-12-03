"""
.. _orientation_widget_example:

Orientation Widget
~~~~~~~~~~~~~~~~~~

Use a orientation widget.

"""

import pyvista as pv
import vtk

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
