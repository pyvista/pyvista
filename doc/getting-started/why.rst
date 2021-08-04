Why PyVista?
============

VTK is an excellent visualization toolkit, and with Python bindings it
should be able to combine the speed of C++ with the rapid prototyping
of Python.  However, despite this VTK code programmed in Python
generally looks the same as its C++ counterpart.  This module seeks to
simplify mesh creation and plotting without losing functionality.

Compare two approaches for loading and plotting a surface mesh from a
file:


Plotting a Mesh using Python's VTK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using this `example <http://www.vtk.org/Wiki/VTK/Examples/Python/STLReader>`_,
loading and plotting an STL file requires a lot of code when using only the
``vtk`` module.

+----------------------------------------------+-------------------------------------------+
| Read a stl file using ``vtk``                | Read a stl file using ``pyvista``         |
+==============================================+===========================================+
| .. code:: python                             | .. code:: python                          |
|                                              |                                           |
|     import vtk                               |     import pyvista                        |
|                                              |                                           |
|     # create reader                          |     mesh = pyvista.read('myfile.stl')     |
|     reader = vtk.vtkSTLReader()              |     mesh.plot()                           |
|     reader.SetFileName("myfile.stl")         |                                           |
|                                              |                                           |
|     mapper = vtk.vtkPolyDataMapper()         |                                           |
|     output_port = reader.GetOutputPort()     |                                           |
|     mapper.SetInputConnection(output_port)   |                                           |
|                                              |                                           |
|     # create actor                           |                                           |
|     actor = vtk.vtkActor()                   |                                           |
|     actor.SetMapper(mapper)                  |                                           |
|                                              |                                           |
|     # Create a rendering window and renderer |                                           |
|     ren = vtk.vtkRenderer()                  |                                           |
|     renWin = vtk.vtkRenderWindow()           |                                           |
|     renWin.AddRenderer(ren)                  |                                           |
|                                              |                                           |
|     # Create a renderwindowinteractor        |                                           |
|     iren = vtk.vtkRenderWindowInteractor()   |                                           |
|     iren.SetRenderWindow(renWin)             |                                           |
|                                              |                                           |
|     # Assign actor to the renderer           |                                           |
|     ren.AddActor(actor)                      |                                           |
|                                              |                                           |
|     # Enable user interface interactor       |                                           |
|     iren.Initialize()                        |                                           |
|     renWin.Render()                          |                                           |
|     iren.Start()                             |                                           |
|                                              |                                           |
|     # clean up objects                       |                                           |
|     del iren                                 |                                           |
|     del renWin                               |                                           |
+----------------------------------------------+-------------------------------------------+

The PyVista data model and API allows you to rapidly load meshes and
handles much of the "grunt work" of setting up plots, connecting
classes and pipelines, and cleaning up plotting windows.  It does this
by exposing a simplified, but functional, interface to VTK's classes.

In :func:`pyvista.read`, PyVista automatically determines the correct
file reader based on the file extension and returns a DataSet object.
This dataset object contains all the methods that are available to a
:class:`pyvista.PolyData` class, including the :func:`pyvista.plot`
method, allowing you to instantly generate a plot of the mesh.
Garbage collection is taken care of automatically and the renderer is
cleaned up after the user closes the plotting window.



Advanced Plotting with Numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyVista is designed around the numpy data model.  When combined with
numpy, you can make some truly spectacular plots:

.. jupyter-execute::
   :hide-code:

   import pyvista
   pyvista.set_jupyter_backend('pythreejs')
   pyvista.global_theme.background = 'white'
   pyvista.global_theme.window_size = [600, 400]
   pyvista.global_theme.axes.show = False
   pyvista.global_theme.antialiasing = True
   pyvista.global_theme.show_scalar_bar = False

.. jupyter-execute::

    import pyvista
    import numpy as np

    # Make a grid
    x, y, z = np.meshgrid(np.linspace(-5, 5, 20),
                          np.linspace(-5, 5, 20),
                          np.linspace(-5, 5, 5))

    points = np.empty((x.size, 3))
    points[:, 0] = x.ravel('F')
    points[:, 1] = y.ravel('F')
    points[:, 2] = z.ravel('F')

    # Compute a direction for the vector field
    direction = np.sin(points)**3

    # plot using the plotting class
    pl = pyvista.Plotter()
    pl.add_arrows(points, direction, 0.5)
    pl.show()

While not everything can be simplified without losing functionality,
many of the objects can.  For example, triangular surface meshes in
VTK can be subdivided but every other object in VTK cannot.  It then
makes sense that a subdivided method be added to the existing
triangular surface mesh.  That way, subdivision can be performed with:

.. jupyter-execute::

    from pyvista import examples
    mesh = examples.load_ant()
    submesh = mesh.subdivide(2, 'linear')
    submesh.plot(show_edges=True)

Additionally, the docstrings for all methods in PyVista are intended
to be used within interactive coding sessions. This allows users to
use sophisticated processing routines on the fly with immediate access
to a description of how to use those methods:

.. figure:: ../images/gifs/documentation.gif
