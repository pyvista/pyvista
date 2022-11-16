"""
Plotting with VTK Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a ``vtkAlgorithm`` to the ``Plotter`` for dynamic visualizations.

A ``vtkAlgorithm`` is the superclass for all sources, filters, and sinks
in VTK. It defines a generalized interface for executing data processing
algorithms. Pipeline connections are associated with input and output
ports that are independent of the type of data passing through the
connections.

We can connect the output port of a ``vtkAlgorithm`` to PyVista's rendering
pipeline when adding data to the scene through methods like ``add_mesh``.

This example will walk through using some ``vtkAlgorithm``'s directly and
passing them to PyVista for dynamic visualization.
"""
import vtk

import pyvista as pv
from pyvista import examples

###############################################################################
# Use ``vtkConeSource`` as a source algorithm. This source will dynamically
# create a cone object depending on the class's parameters. In this example,
# we will connect a callback to set the cone source algorithm's resolution
# via ``vtkConeSource.SetResolution``
algo = vtk.vtkConeSource()


def update_resolution(value):
    """Callback to easily set the resolution of the cone generator."""
    res = int(value)
    algo.SetResolution(res)
    return


###############################################################################
# Pass the ``vtkConeSource`` (a ``vtkAlgorithm`` subclass) directly to the
# Plotter and connect a slider widget to our callback to adjust the resolution
p = pv.Plotter()
p.add_mesh(algo, color='red')
p.add_slider_widget(update_resolution, [5, 100], title='Resolution')
p.show()

###############################################################################
# Or we can use ``vtkRegularPolygonSource``
poly_source = vtk.vtkRegularPolygonSource()
poly_source.GeneratePolygonOff()
poly_source.SetRadius(5.0)
poly_source.SetCenter(0.0, 0.0, 0.0)


def update_n_sides(value):
    """Callback to easily set the number of sides."""
    res = int(value)
    poly_source.SetNumberOfSides(res)
    return


p = pv.Plotter()
p.add_mesh_clip_box(poly_source, color='red')
p.add_slider_widget(update_n_sides, [3, 25], title='N Sides')
p.view_xy()
p.show()


###############################################################################
# Filter Pipeline
# +++++++++++++++
# We can do this with any ``vtkAlgorithm`` subclass for dynamically generating
# or filtering data. Here is an example of executing a pipeline of VTK filters
# together.

# Source mesh object (static)
mesh = examples.download_bunny_coarse()

# Initialize VTK algorithm to modify dynamically
splatter = vtk.vtkGaussianSplatter()

# Pass PyVista object as input to VTK
splatter.SetInputData(mesh)

# Set parameters of splatter filter
n = 200
splatter.SetSampleDimensions(n, n, n)
splatter.SetRadius(0.02)
splatter.SetExponentFactor(-10)
splatter.SetEccentricity(2)
splatter.Update()

# Pipe splatter filter into a contour filter
contour = vtk.vtkContourFilter()
contour.SetInputConnection(splatter.GetOutputPort())
contour.SetInputArrayToProcess(0, 0, 0, 0, 'SplatterValues')
contour.SetNumberOfContours(1)
contour.SetValue(0, 0.95 * splatter.GetRadius())

# Use PyVista to plot output of contour filter
p = pv.Plotter(notebook=0)
p.add_mesh(mesh, style='wireframe')
p.add_mesh(contour, color=True)
p.add_slider_widget(splatter.SetRadius, [0.01, 0.05])
p.show()
