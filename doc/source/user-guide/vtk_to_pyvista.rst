.. _vtk_to_pyvista_docs:


Transitioning from VTK to PyVista
=================================
VTK is primarily developed in C++ and uses chained setter and getter
commands to access data. Instead, PyVista wraps the VTK data types
into numpy arrays so that users can benefit from its bracket syntax
and fancy indexing. This section demonstrates the difference between
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

The creation of a simple `vtk.vtkImageData`_ dataset involves a significant
amount of boilerplate code. However, PyVista provides more concise and
"Pythonic" syntax. The equivalent code in PyVista is:

.. code:: python

   >>> import pyvista
   >>> import numpy as np

   Use the meshgrid function to create 2D "grids" of the x and y values.
   This section effectively replaces the vtkDoubleArray.

   >>> xi = np.arange(300)
   >>> x, y = np.meshgrid(xi, xi)
   >>> values = 127.5 + (1.0 + np.sin(x/25.0)*np.cos(y/25.0))

   Create the grid. Note how the values must use Fortran ordering.

   >>> grid = pyvista.UniformGrid(dimensions=(300, 300, 1))
   >>> grid.point_data["values"] = values.flatten(order="F")

PyVista simplifies several steps in this process:

#. Combines the dimensionality of the data (in the shape of the
   :class:`numpy.ndarray`) with the values of the data in one line. In
   contrast, VTK uses "tuples" to describe the shape of the data (where it sits
   in space) and "components" to describe the type of data (1 = scalars/scalar
   fields, 2 = vectors/vector fields, n = tensors/tensor fields). With PyVista,
   shape and values are stored concretely in one variable.

#. :class:`pyvista.UniformGrid` is a wrapper around `vtk.vtkImageData`_ and
   functions as a container for evenly spaced points. Despite the name, your
   data does not have to be an "image" to be used with
   `vtk.vtkImageData`_. Instead, the dataset can contain values that are spaced
   evenly apart, similar to pixels in an image.

   Since PyVista knows that the container is for uniformly spaced data, it sets
   the origin and spacing to ``(0, 0, 0)`` and ``(1, 1, 1)`` by default. This
   is one of the advantages of PyVista and Python. You don't need to know
   everything about the VTK library upfront to get started. You can start
   easily and dive deeper once you become more familiar with it. For example,
   changing the origin and spacing is as simple as:

   .. code:: python

      >>> grid.origin = (10, 20, 10)
      >>> grid.spacing = (2, 3, 5)

#. The name for the :attr:`point_array <pyvista.point_array>` is given
   directly in dictionary-style fashion. Also, since VTK stores data
   on the heap (linear segments of RAM; a C++ concept), the
   data must be flattened and put in FORTRAN ordering (which controls
   how multidimensional data is laid out in physically 1D memory; numpy
   uses "C"-style memory layout by default). This is why in the earlier
   example, the first argument to ``SetValue()`` was written as
   ``x*300 + y``. Here, numpy takes care of this quite nicely
   and it's made more explicit in the code, following the Python best
   practice of "Explicit is better than implicit."

Finally, with PyVista, each geometry class contains methods that allow
you to immediately plot the mesh without also setting up the plot.
For example, in VTK you would have to do:

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

However, with PyVista you only need:

.. code:: python

   grid.plot(cpos='xy', show_scalar_bar=False, cmap='coolwarm')

..
   This is here to generate a plot. Everything is repeated since
   jupyter-execute doesn't allow for plain text between command blocks.

.. jupyter-execute::
   :hide-code:

   import pyvista as pv
   pv.set_plot_theme('document')
   pv.set_jupyter_backend('static')
   import numpy as np
   xi = np.arange(300)
   x, y = np.meshgrid(xi, xi)
   values = 127.5 + (1.0 + np.sin(x/25.0)*np.cos(y/25.0))
   grid = pv.UniformGrid(dimensions=(300, 300, 1))
   grid.point_data["values"] = values.flatten(order="F")
   grid.plot(cpos='xy', show_scalar_bar=False, cmap='coolwarm')


.. _vtk.vtkImageData: https://vtk.org/doc/nightly/html/classvtkImageData.html


Construction of a ``PointSet``
------------------------------
PyVista heavily relies on NumPy to efficiently allocate and access
VTK's C arrays. For example, to create an array of points within VTK
one would normally loop through all the points of a list and supply
that to a  `vtkPoints`_ class. For example:

.. jupyter-execute::

   >>> import vtk
   >>> vtk_array = vtk.vtkDoubleArray()
   >>> vtk_array.SetNumberOfComponents(3)
   >>> vtk_array.SetNumberOfValues(9)
   >>> vtk_array.SetValue(0, 0)
   >>> vtk_array.SetValue(1, 0)
   >>> vtk_array.SetValue(2, 0)
   >>> vtk_array.SetValue(3, 1)
   >>> vtk_array.SetValue(4, 0)
   >>> vtk_array.SetValue(5, 0)
   >>> vtk_array.SetValue(6, 0.5)
   >>> vtk_array.SetValue(7, 0.667)
   >>> vtk_array.SetValue(8, 0)
   >>> vtk_points = vtk.vtkPoints()
   >>> vtk_points.SetData(vtk_array)
   >>> print(vtk_points)

To do the same within PyVista, you simply need to create a NumPy array:

.. jupyter-execute::

   >>> import numpy as np
   >>> np_points = np.array([[0, 0, 0],
   ...                       [1, 0, 0],
   ...                       [0.5, 0.667, 0]])

.. note::
   You can use :func:`pyvista.vtk_points` to construct a `vtkPoints`_
   object, but this is unnecessary in almost all situations.

Since the end goal is to construct a :class:`pyvista.DataSet
<pyvista.core.dataset.DataSet>`, you would simply pass the
``np_points`` array to the :class:`pyvista.PolyData` constructor:

.. jupyter-execute::

   >>> import pyvista
   >>> poly_data = pyvista.PolyData(np_points)

Whereas in VTK you would have to do:

.. jupyter-execute::

   >>> vtk_poly_data = vtk.vtkPolyData()
   >>> vtk_poly_data.SetPoints(vtk_points)

The same goes with assigning face or cell connectivity/topology. With
VTK you would normally have to loop using ``InsertNextCell`` and
``InsertCellPoint``. For example, to create a single cell
(triangle) and then assign it to `vtkPolyData`_:

.. jupyter-execute::

   >>> cell_arr = vtk.vtkCellArray()
   >>> cell_arr.InsertNextCell(3)
   >>> cell_arr.InsertCellPoint(0)
   >>> cell_arr.InsertCellPoint(1)
   >>> cell_arr.InsertCellPoint(2)
   >>> vtk_poly_data.SetPolys(cell_arr)

With PyVista, you can assign the faces of a :class:`pyvista.PolyData` directly
in the constructor and then access or modify them using the :attr:`faces
<pyvista.PolyData.faces>` attribute.

.. jupyter-execute::

   >>> faces = np.array([3, 0, 1, 2])
   >>> poly_data = pyvista.PolyData(np_points, faces)
   >>> poly_data.faces

.. _vtk_vs_pyvista_object_repr:

Object representation
---------------------
Both VTK and PyVista provide representations for their objects.

VTK provides a verbose representation (useful for debugging) of their data types
that can be accessed via :func:`print`, as the ``__repr__``
(unlike ``__str__``) only provides minimal information about each object:

.. jupyter-execute::

   >>> print(vtk_poly_data)

PyVista chooses to show minimal data in the :func:`repr`, preferring
explicit attribute access on meshes for the bulk of attributes.
For example:

.. jupyter-execute::

   >>> poly_data

This representation shows the following information about the mesh:

* Number of cells :attr:`n_cells <pyvista.DataSet.n_cells>`
* Number of points :attr:`n_points <pyvista.DataSet.n_points>`
* Bounds of the mesh :attr:`bounds <pyvista.DataSet.bounds>`
* Number of data arrays :attr:`n_arrays <pyvista.DataSet.n_arrays>`

All other attributes like :attr:`lines <pyvista.PolyData.lines>`,
:attr:`point_data <pyvista.DataSet.point_data>`, or
:attr:`cell_data <pyvista.DataSet.cell_data>` can be
accessed directly from the object. This approach was chosen to allow
for a brief summary showing key parts of the :class:`DataSet
<pyvista.DataSet>` without overwhelming the user.

Tradeoffs
---------
Although PyVista simplifies many features, not everything can be simplified
without sacrificing functionality or performance.

For instance, the :class:`collision <pyvista.PolyDataFilters.collision>` filter
shows how to compute the collision between two meshes:

.. jupyter-execute::
   :hide-code:

   # must have this here as our global backend may not be static
   import pyvista
   pyvista.set_jupyter_backend('pythreejs')
   pyvista.global_theme.window_size = [600, 400]
   pyvista.global_theme.anti_aliasing = 'fxaa'


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

Under the hood, the collision filter detects mesh collisions using
oriented bounding box (OBB) trees. For a single collision, this filter
is as performant as the VTK counterpart, but when computing multiple
collisions with the same meshes, as in the :ref:`collision_example`
example, it is more efficient to use the `vtkCollisionDetectionFilter
<https://vtk.org/doc/nightly/html/classvtkCollisionDetectionFilter.html>`_,
as the OBB tree is computed once for each mesh. In most cases, pure
PyVista is sufficient for most data science, but there are times when
you may want to use VTK classes directly.

Note that nothing stops you from using VTK classes and then wrapping
the output with PyVista. For example:

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
the flexibility of PyVista and the raw power of VTK.

.. note::
   You can use :func:`pyvista.Polygon` for a one line replacement of
   the above VTK code.


.. _vtkDataArray: https://vtk.org/doc/nightly/html/classvtkDataArray.html
.. _vtkPolyData: https://vtk.org/doc/nightly/html/classvtkPolyData.html
.. _vtkImageData: https://vtk.org/doc/nightly/html/classvtkImageData.html
.. _vtkpoints: https://vtk.org/doc/nightly/html/classvtkPoints.html
