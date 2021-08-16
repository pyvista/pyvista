.. _pyvista_data_model:


PyVista Data Model
==================
This section of the user guide explains in detail how to construct
meshes from scratch and to utilize the underlying VTK data model but
using the PyVista framework.  Many of our :ref:`ref_examples` simply
load data from files, but don't explain how to construct meshes or
place data within datasets.

.. note::
   Though the following documentation section references VTK, it does
   not require that you have knowledge of VTK.  For those who wish to
   see a detailed comparison to VTK or translate code written for the
   Python bindings of VTK to PyVista, please see
   :ref:`pyvista_to_vtk_docs`.

For a more general description of our api, see :ref:`what_is_a_mesh`.


The PyVista DataSet
-------------------
To visualize data in VTK or PyVista, two pieces of information are
required: the data's geometry, which describes where the data is
positioned in space and what is values are, and its topology, which
describes how points in the dataset are connected to one another.

At the top level, we have `vtkDataObject`_, which are just "blobs" of
data without geometry or topology. These contain arrays of
`vtkFieldData`_. Under this are `vtkDataSet`_, which add geometry and
topolgy to `vtkDataObject`_. Associated with every point and cell in
the dataset is a specific value. Since these values must be positioned
and connected in space, they are held in the `vtkDataArray`_ class,
which are simply memory buffers on the heap. In PyVista, 99% of the
time we interact with `vtkDataSet`_ objects rather than with
`vtkDataObject`_ objects. PyVista uses the same data types as VTK, but
structures them in a more pythonic manner for ease of use.

If you'd like a background for how VTK structures its data, see
`Introduction to VTK in Python by Kitware
<https://vimeo.com/32232190>`_, as well as the numerous code examples
on `Kitware's GitHub site
<https://kitware.github.io/vtk-examples/site/>`_. An excellent
introduction to mathematical concept relevant to 3D modeling in
general implemented in VTK is provided by the `Discrete Differential
Geometry YouTube Series
<https://www.youtube.com/playlist?list=PL9_jI1bdZmz0hIrNCMQW1YmZysAiIYSSS>`_
by Prof. Keenan Crane at Carnegie Melon. The concepts taught here
will help improve your understanding of why data sets are structured
the way they are in libraries like VTK.

At the most fundamental level, all PyVista geometry classes inherit
from the :ref:`ref_dataset` class. A dataset has geometry, topology,
and attributes describing that geometry in the form of point, cell, or
field arrays.

Geometry in PyVista is represented as points and cells.  For example,
consider a single cell within a :class:`pyvista.PolyData`:

.. jupyter-execute::
   :hide-code:

   import pyvista
   pyvista.set_plot_theme('document')
   pyvista.set_jupyter_backend('static')
   points = [[.2, 0, 0], [1.3, 0, 0], [1, 1.2, 0], [0, 1, 0]]
   cells = [4, 0, 1, 2, 3]
   mesh = pyvista.PolyData(points, cells)

   pl = pyvista.Plotter()
   pl.add_mesh(mesh, show_edges=False)
   pl.add_mesh(mesh.extract_feature_edges(), line_width=5, color='k')
   pl.add_point_labels(mesh.points, [f'Point {i}' for i in range(4)], font_size=20, point_size=20)
   pl.add_point_labels(mesh.center, ['Cell'], font_size=28)
   pl.camera_position = 'xy'
   pl.show()

We would need a way to describe the position of each of these points
in space, but we're limited to expressing the values themselves as
we've done above (lists of arrays with indices). VTK (and hence
PyVista) have multiple classes that represent different data
shapes. The most important dataset classes are shown below:

.. jupyter-execute::
   :hide-code:

   from pyvista import demos
   demos.plot_datasets()

Here, the above datasets are ordered from most (5) to least complex
(1). That is, every dataset can be represented as an
:class:`pyvista.UnstructuredGrid`, but the
:class:`pyvista.UnstructuredGrid` class takes the most amount of
memory to store since they must account for every individual point and
cell . On the other hand, since `vtkImageData`_
(:class:`pyvista.UniformGrid`) is uniformly spaced, a few integers and
floats can describe the shape, so it takes the least amount of memory
to store."

This is because in :class:`pyvista.PolyData` or
:class:`pyvista.UnstructuredGrid`, points and cells must be explicitly
defined.  In other data types, such as :class:`pyvista.UniformGrid`,
the cells (and even points) are defined as a emergent property based
on the dimensionality of the grid.

To see this in practice, let's create the simplest surface represented
as a :class:`pyvista.PolyData`. First, we need to define our points.


Points and Arrays in PyVista
----------------------------
There are a variety of ways to create points within PyVista, and this section shows how to efficiently create an array of points by either:

* Wrapping a VTK array
* Using a :class:`numpy.ndarray` array
* Or just using a :class:`list`

PyVista provides pythonic methods for all three approaches so you can
choose whatever is most efficient for you. If you're comfortable with
the VTK API, you can choose to wrap VTK arrays, but you may find that
using :class:`numpy.ndarray` is more convenient and avoids the looping
overhead in Python.

Wrapping a VTK Array
~~~~~~~~~~~~~~~~~~~~
Let's define points of a triangle. Using the VTK API, this can be
done with:

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
   >>> print(vtk_array)

PyVista supports creating objects directly from the `vtkDataArray`_
class, but there's a better, and more pythonic alternative by using
:class:`numpy.ndarray`.


Using NumPy with PyVista
~~~~~~~~~~~~~~~~~~~~~~~~
However, there's no reason to do this since Python already has the
excellent C array library `NumPy <https://numpy.org/>`_. You could
more create a points array with:

.. jupyter-execute::

   >>> import numpy as np
   >>> np_points = np.array([[0, 0, 0],
   ...                       [1, 0, 0],
   ...                       [0.5, 0.667, 0]])
   >>> np_points

We use a :class:`numpy.ndarray` here so that PyVista directly "point"
the underlying C array to VTK. VTK already has APIs to directly read
in the C arrays from ``numpy``, and since VTK is written in C++,
everything from Python that is transferred over to VTK needs to be in a
format that VTK can process.

Should you wish to use VTK objects within PyVista, you can still do
this. In fact, using :func:`pyvista.wrap`, you can even get a numpy-like
representation of the data. For example:

.. jupyter-execute::

   >>> import pyvista
   >>> wrapped = pyvista.wrap(vtk_array)
   >>> wrapped

Note that when wrapping the underlying VTK array, we actually perform
a shallow copy of the data. In other words, we pass the pointer from
the underlying C array to the numpy :class:`numpy.ndarray`, meaning
that the two arrays are now efficiently linked. This means that we
can change the array using numpy array indexing and have it modified
on the "VTK side".

.. jupyter-execute::

   >>> wrapped[0, 0] = 10
   >>> vtk_array.GetValue(0)

Or we can change the value from the VTK array and see it reflected in
the numpy wrapped array. Let's change the value back:

.. jupyter-execute::

   >>> vtk_array.SetValue(0, 0)
   >>> wrapped[0, 0]


Using Python Lists or Tuples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyVista supports the use of Python sequences (i.e. :class:`list` or
:class:`tuple`, and you could define a your points using a nested list
of lists via:

.. jupyter-execute::

   >>> points = [[0, 0, 0],
   ...           [1, 0, 0],
   ...           [0.5, 0.667, 0]]

When used in the context of :class:`pyvista.PolyData` to create the
mesh, this list will automatically be wrapped using numpy and then
passed to VTK. This avoids any looping overhead and while still
allowing you to use native python classes.

Finally, let's show how we can use these three objects in the context
of a PyVista geometry class. Here, we create a simple point mesh
containing just the three points:

.. jupyter-execute::
   
   >>> from_vtk = pyvista.PolyData(vtk_array)
   >>> from_np = pyvista.PolyData(np_points)
   >>> from_list = pyvista.PolyData(points)

These point meshes all contain three points and are effectively
identical. Let's show this by accessing the underlying points array
from the mesh, which is represented as a :class:`pyvista.pyvista_ndarray`

.. jupyter-execute::

   >>> from_vtk.points

And show that these are all identical

.. jupyter-execute::

   >>> assert np.array_equal(from_vtk.points, from_np.points)
   >>> assert np.array_equal(from_vtk.points, from_list.points)
   >>> assert np.array_equal(from_np.points, from_list.points)

Finally, let's plot this (very) simple example using PyVista's
:func:`pyvista.plot` method. Let's make this a full example so you
can see the entire process.

.. pyvista-plot::
   :context:

   >>> import pyvista
   >>> points = [[0, 0, 0],
   ...           [1, 0, 0],
   ...           [0.5, 0.667, 0]]
   >>> mesh = pyvista.PolyData(points)
   >>> mesh.plot(show_bounds=True, cpos='xy', point_size=20)

We'll get into PyVista's data classes and attributes later, but for
now we've shown how create a simple geometry containing just points.
To create a surface, we must specify the connectivity of the geometry,
and to do that we need to specify the cells (or faces) of this surface.


Geometry and Mesh Connectivity/Topology within PyVista
------------------------------------------------------
With our previous example, we defined our "mesh" as three disconnected
points. While this is useful for representing "point clouds", if we
want to create a surface, we have to describe the connectivity of the
mesh. To do this, let's define a single cell composed of three points
in the same order as we defined earlier.

.. jupyter-execute::

   >>> cells = [3, 0, 1, 2]

.. note::
   Observe how we had insert a leading ``3`` to tell VTK that our face
   will contain three points. In our :class:`pyvista.PolyData` VTK
   doesn't assume that faces always contain three points, so we have
   to define that. This actually gives us the flexibility to define
   as many (or as few as one) points per cell as we wish.


Now we have all the necessary pieces to assemble an instance of
:class:`pyvista.PolyData` that contains a single triangle. To do
this, we simply provide the ``points`` and ``cells`` to the
constructor of a :class:`pyvista.PolyData`. We can see from the
representation that this geometry contains three points and one cell

.. jupyter-execute::

   >>> mesh = pyvista.PolyData(points, cells)
   >>> mesh

Let's also plot this:

.. pyvista-plot::
   :context:

   >>> mesh = pyvista.PolyData(points, [3, 0, 1, 2])
   >>> mesh.plot(cpos='xy', show_edges=True)

While we're at it, let's annotate this plot to describe this mesh.

.. pyvista-plot::
   :context:

   >>> pl = pyvista.Plotter()
   >>> pl.add_mesh(mesh, show_edges=True, line_width=5)
   >>> pl.add_point_labels(mesh.points, [f'Point {i}' for i in range(3)], 
   ...                     font_size=20, point_size=20)
   >>> pl.add_point_labels([0.43, 0.2, 0], ['Cell 0'], font_size=20)
   >>> pl.camera_position = 'xy'
   >>> pl.show()

You can clearly see how the polygon is created based on the
connectivity of the points.

This instance has several attributes to access the underlying data of
the mesh. For example, if you wish to access or modify the points of
the mesh, you can simply access the points attribute with
:attr:`points <pyvista.core.dataset.DataSet.points>`.

.. jupyter-execute::

   >>> mesh.points

The connectivity can also be accessed from the :attr:`faces <pyvista.PolyData.faces>`
attribute with:

.. jupyter-execute::

   >>> mesh.faces

Or we could simply get the representation of the mesh with:

.. jupyter-execute::

   >>> mesh

In this representation we see:

* Number of points :attr:`n_points <pyvista.core.dataset.DataSet.n_points>`
* Number of cells :attr:`n_points <pyvista.core.dataset.DataSet.n_cells>`
* Bounds of the mesh :attr:`bounds <pyvista.core.dataset.DataSet.bounds>`
* Number of data arrays :attr:`n_arrays <pyvista.core.dataset.DataSet.n_arrays>`

This is vastly different from the output from VTK.  See
:ref:`pyvista_vs_vtk_object_repr` for the comparison between the two
representations.

This mesh contains no data arrays consists only of geometry.  This
makes it useful for plotting just the geometry of the mesh, but
datasets often contain more than just geometry.  For example:

* An electrical field computed from a changing magnetic field
* Vector field of blood flow through artery
* Surface stresses from a structural finite element analysis
* Mineral deposits from geophysics
* Weather patterns as a vector field or surface data.

While each one of these datasets could be represented as a different
geometry class, they would all contain point, cell, or field data that
explains the value of the data at a certain location within the
geometry.


Data Arrays
-----------
Each :class:`pyvista.DataSet <pyvista.core.dataset.DataSet>` contains
attributes that allow you to access the underlying numeric data.  This
numerical data may be associated with the :attr:`points
<pyvista.core.dataset.DataSet.points>`, :attr:`cells
<pyvista.core.dataset.DataSet.cells>`, or not associated with points
or cells and attached to the mesh in general.

To illustrate data arrays within PyVista, let's first construct a
slightly more complex mesh than our previous example.  Here, we create
a simple mesh containing four equal cells by starting with a
:class:`pyvista.UniformGrid` and then casting it to a
:class:`pyvista.UnstructuredGrid` with with :func:`cast_to_unstructured_grid <pyvista.core.dataset.DataSet.cast_to_unstructured_grid`.

.. jupyter-execute::

   >>> grid = pyvista.UniformGrid((3, 3, 1))
   >>> ugrid = grid.cast_to_unstructured_grid()
   >>> ugrid

Let's also plot this basic mesh:

.. pyvista-plot::
   :context:
   :include-source: False

   >>> grid = pyvista.UniformGrid((3, 3, 1))
   >>> ugrid = grid.cast_to_unstructured_grid()

.. pyvista-plot::
   :context:

   >>> pl = pyvista.Plotter()
   >>> pl.add_mesh(ugrid, show_edges=True, line_width=5)
   >>> point_labels = [f'Point {i}' for i in range(ugrid.n_points)]
   >>> pl.add_point_labels(ugrid.points, point_labels,
   ...                     font_size=25, point_size=20)
   >>> cell_labels = [f'Cell {i}' for i in range(ugrid.n_cells)]
   >>> pl.add_point_labels(ugrid.cell_centers(), cell_labels, font_size=25)
   >>> pl.camera_position = 'xy'
   >>> pl.show()

Now that we have a simple mesh to work with, we can start assigning it
data.  There are two main types of data that can be associated with
mesh, scalar data and vector data. Scalar data is single or
multi-component data that is non directional and may include values
like temperature, or in the case of multi-component data, RGBA values.
Vector data contains magnitude and direction and is represented as
arrays containing three components.

When plotting, we can easily display scalar data, but this data must
be "associated" with either points or cells.  For example, we may wish
to assign values to the cells of our example mesh, which we can do by
accessing the `attr:`cell_arrays <pyvista.DataSet.cell_arrays>`
attribute of our mesh.


Cell Arrays
~~~~~~~~~~~
The easiest way to add scalar data to a :class:`DataSet
<pyvista.core.dataset.DataSet>` is to use the ``[]`` operator.
Continuing with our example above, let's assign each cell a single
integer.  We can do this using a python :class:`list` and making it
the same length as the number of cells in the
:class:`pyvista.UnstructuredGrid`.  Here we create the list, add it to
the `attr:`cell_arrays <pyvista.DataSet.cell_arrays>`, and then access
it using the ``[]`` operator.

.. jupyter-execute::

   >>> simple_list = range(ugrid.n_cells)
   >>> ugrid.cell_arrays['my-data'] = simple_list
   >>> ugrid.cell_arrays['my-data']

Note how we are returned a :class:`pyvista.pyvista_ndarray`.  Since
VTK requires C arrays, PyVista will internally wrap or convert all
inputs to C arrays.  We can then plot this with:

.. pyvista-plot::
   :context:
   :include-source: False

   >>> ugrid.cell_arrays['my-data'] = range(ugrid.n_cells)

.. pyvista-plot::
   :context:

   >>> ugrid.plot(cpos='xy', show_edges=True)

Note how we did not have to specify which cell data to plot as the
``[]`` operator automatically sets the active scalars:

.. jupyter-execute::

   >>> ugrid.cell_arrays

We can also add 

   >>> pl = pyvista.Plotter()
   >>> pl.add_mesh(ugrid, show_edges=True, line_width=5)
   >>> point_labels = [f'Point {i}' for i in range(ugrid.n_points)]
   >>> pl.add_point_labels(ugrid.points, point_labels,
   ...                     font_size=25, point_size=20)
   >>> cell_labels = [f'Cell {i}' for i in range(ugrid.n_cells)]
   >>> pl.add_point_labels(ugrid.cell_centers(), cell_labels, font_size=25)
   >>> pl.camera_position = 'xy'
   >>> pl.show()


Point Arrays
~~~~~~~~~~~~




Field Arrays
~~~~~~~~~~~~



.. _vtkDataArray: https://vtk.org/doc/nightly/html/classvtkDataArray.html
.. _vtkDataSet: https://vtk.org/doc/nightly/html/classvtkDataSet.html
.. _vtkFieldData: https://vtk.org/doc/nightly/html/classvtkFieldData.html
.. _vtkDataObject: https://vtk.org/doc/nightly/html/classvtkDataObject.html
.. _vtk.vtkPolyData: https://vtk.org/doc/nightly/html/classvtkPolyData.html
.. _vtk.UnstructuredGrid: https://vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html
.. _vtk.vtkStructuredGrid: https://vtk.org/doc/nightly/html/classvtkStructuredGrid.html
.. _vtk.vtkRectilinearGrid: https://vtk.org/doc/nightly/html/classvtkRectilinearGrid.html
.. _vtkImageData: https://vtk.org/doc/nightly/html/classvtkImageData.html
.. _vtk.vtkMultiBlockDataSet: https://vtk.org/doc/nightly/html/classvtkMultiBlockDataSet.html

.. _cast_to_unstructured_grid: :func:`cast_to_unstructured_grid <pyvista.core.dataset.DataSet.cast_to_unstructured_grid>`
