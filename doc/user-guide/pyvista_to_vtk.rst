.. _pyvista_to_vtk_docs:


Transitioning from VTK to PyVista
=================================
VTK is primarily developed in C++ and uses chained setter and getter
commands to access data. Instead, PyVista wraps the VTK data types
into numpy arrays so that users can benefit from its bracket syntax
and fancy indexing.  This section demonstrates the difference between
the two approaches in a series of examples.

For example, to hard-code points for a `vtk.vtkImageData`_ data
structure using VTK Python's bindings, one would write the following:

.. code:: python

   >>> import vtk
   >>> from math import cos, sin

   Create (x, y) points for a 300x300 image dataset

   >>> points = vtk.vtkDoubleArray()
   >>> points.SetName("points")
   >>> points.SetNumberOfComponents(1)
   >>> points.SetNumberOfTuples(300*300)

   >>> for x in range(300):
   ...     for y in range(300):
   ...         points.SetValue(x*300 + y, 127.5 + (1.0 + sin(x/25.0)*cos(y/25.0)))

   Create the image structure

   >>> image_data = vtk.vtkImageData()
   >>> image_data.SetOrigin(0, 0, 0)
   >>> image_data.SetSpacing(1, 1, 1)
   >>> image_data.SetDimensions(300, 300, 1)

   Assign the points to the image

   >>> image_data.GetPointData().SetScalars(points)

As you can see, there is quite a bit of boiler-plate that goes into
the creation of a simple `vtk.vtkImageData`_ dataset, PyVista provides
a much cleaner syntax that is both more readable and intuitive. The
equivalent code in pyvista is:


.. code:: python

   >>> import pyvista
   >>> import numpy as np

   Use the meshgrid method to create 2D "grids" of the x and y values.
   This section effectively replaces ``vtkDoubleArray``

   >>> xi = np.arange(300)
   >>> x, y = np.meshgrid(xi, xi)
   >>> values = 127.5 + (1.0 + np.sin(x/25.0)*np.cos(y/25.0))

   Create the grid.  Note how the values must use FORTRAN ordering.

   >>> grid = pyvista.UniformGrid((300, 300, 1))
   >>> grid.point_arrays["values"] = values.flatten(order="F")

Here, PyVista has done several things for us:

#. PyVista combines the dimensionality of the data (in the shape of
   the :class:`numpy.ndarray`) with the values of the data in one line. VTK uses
   "tuples" to describe the shape of the data (where it sits in space)
   and "components" to describe the type of data (1 = scalars/scalar
   fields, 2 = vectors/vector fields, n = tensors/tensor
   fields). Here, shape and values are stored concretely in one
   variable.

#. :class:`pyvista.UniformGrid` wraps `vtk.vtkImageData`_, just with a
   better name; they are both containers of evenly spaced points. Your
   data does not have to be an "image" to use it with vtk.vtkImageData;
   rather, like images, values in the dataset are evenly spaced apart
   like pixels in an image.

   Furthermore, since we know the container is for uniformly spaced data,
   pyvista sets the origin and spacing by default to ``(0, 0, 0)`` and
   ``(1, 1, 1)``. This is another great thing about PyVista and python!
   Rather than having to know everything about the VTK library up front,
   you can get started very easily! Once you get more familiar with it
   and need to do something more complex, you can dive deeper. For
   example, changing the origin and spacing is as simple as:

   .. code:: python

      >>> grid.origin = (10, 20, 10)
      >>> grid.spacing = (2, 3, 5)

#. The name for the :attr:`point_array <pyvista.point_array>` is given
   directly in dictionary-style fashion. Also, since VTK stores data
   on the heap (linear segments of RAM memory; a C++ concept), the
   data must be flattened and put in FORTRAN ordering (controls the
   data "endianness"; numpy uses "C"-style endianness by
   default). This is why in our earlier example, the first argument to
   ``SetValue()`` was written as ``x*300 + y``. Here, numpy takes care of
   this for us quite nicely and it's made more explicit in the code,
   following the Python best practice of "Explicit is better than
   implicit".

Finally, with PyVista, each geometry class contains methods that allow
you to immediately plot the mesh without also setting up the plot.
For example, in VTK you would:

.. code:: python

   >>> actor = vtk.vtkImageActor()
   >>> actor.GetMapper().SetInputData(image_data)
   >>> ren = vtk.vtkRenderer()
   >>> renWin = vtk.vtkRenderWindow()
   >>> renWin.AddRenderer(ren)
   >>> renWin.SetWindowName('ReadSTL')
   >>> iren = vtk.vtkRenderWindowInteractor()
   >>> iren.SetRenderWindow(renWin)
   >>> ren.AddActor(actor)
   >>> iren.Initialize()
   >>> renWin.Render()
   >>> iren.Start()

However, with PyVista you simply need:

.. code:: python

   grid.plot(cpos='xy', show_scalar_bar=False, cmap='coolwarm')

..
   This is here so we can generate a plot.  We have to repeat
   everything since jupyter-execute doesn't allow for
   plain text between command blocks.

.. jupyter-execute::
   :hide-code:

   import pyvista as pv
   pv.set_plot_theme('document')
   pv.set_jupyter_backend('static')
   import numpy as np
   xi = np.arange(300)
   x, y = np.meshgrid(xi, xi)
   values = 127.5 + (1.0 + np.sin(x/25.0)*np.cos(y/25.0))
   grid = pv.UniformGrid((300, 300, 1))
   grid.point_arrays["values"] = values.flatten(order="F")
   grid.plot(cpos='xy', show_scalar_bar=False, cmap='coolwarm')


.. _vtk.vtkImageData: https://vtk.org/doc/nightly/html/classvtkImageData.html



Tradeoffs
~~~~~~~~~
While most features can, not everything can be simplified without
losing functionality or performance.

In the :class:`collision <pyvista.PolyDataFilters.collision>` filter,
we demonstrate how to calculate the collision between two meshes.  For
example:

.. jupyter-execute::

   import pyvista

   # create a default sphere and a shifted sphere
   mesh_a = pyvista.Sphere()
   mesh_b = pyvista.Sphere(center=(-0.4, 0, 0))
   out, n_coll = mesh_a.collision(mesh_b, generate_scalars=True, contact_mode=2)

   pl = pyvista.Plotter()
   pl.add_mesh(out)
   pl.add_mesh(mesh_b, style='wireframe', color='k')
   pl.camera_position = 'xy'
   pl.show()

Under the hood, the collision filter detects mesh collisions using a
oriented bounding box (OBB) trees.  For a single collision, this filter
is as performant as the vtk counterpart, but when computing multiple
collisions with the same meshes, as in the :ref:`collision_example`
example, it is more efficient (though less convienent) to use the VTK
underlying `vtkCollisionDetectionFilter
<https://vtk.org/doc/nightly/html/classvtkCollisionDetectionFilter.html>`_,
as the OBB tree is computed once for each mesh.  In most cases, pure
PyVista is sufficient for most data science, but there are times when
you may want to use VTK classes directly.

Note that nothing stops you from using VTK classes and then wrapping
the output with PyVista.  For example:

.. jupyter-execute::
   
   import vtk
   import pyvista

   # Create a circle using vtk
   polygonSource = vtk.vtkRegularPolygonSource()
   polygonSource.GeneratePolygonOff()
   polygonSource.SetNumberOfSides(50)
   polygonSource.SetRadius(5.0)
   polygonSource.SetCenter(0.0, 0.0, 0.0)
   polygonSource.Update()

   # wrap and plot using pyvista
   mesh = pyvista.wrap(polygonSource.GetOutput())
   mesh.plot(line_width=3, cpos='xy', color='k')

In this manner, you can get the "best of both worlds" should you need
the flexibility of PyVista and the functionality of VTK.

.. note::
   You can use :func:`pyvista.Circle` for a one line replacement of
   the above VTK code.
