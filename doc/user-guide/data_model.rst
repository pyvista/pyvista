.. _pyvista_data_model:


PyVista Data Model
==================
PyVista uses the same data types as VTK, but structures them in a more
pythonic manner for ease of use. If you'd like a background for how
VTK structures its data, see `Introduction to VTK in Python by Kitware
<https://vimeo.com/32232190>`_, as well as the numerous code examples
on `Kitware's GitHub site
<https://kitware.github.io/vtk-examples/site/>`_. An excellent
introduction to mathematical concept implemented in VTK is provided by
the `Discrete Differential Geometry YouTube Series
<https://www.youtube.com/playlist?list=PL9_jI1bdZmz0hIrNCMQW1YmZysAiIYSSS>`_
by Prof. Keenan Crane at Carnegie Melon.


PyVista DataSet
===============

At the most fundamental level, all PyVista geometry objects are
:ref:`ref_dataset`.  A dataset is a surface or volume in 3D space
containing points, cells, and attributes describing that geometry.

Geometry in PyVista is represented as points and cells.  In certain
geometry types, such as :class:`pyvista.PolyData` or
:class:`pyvista.UnstructuredGrids`, these cells must be explicitaly
defined.  In other data types, such as :class:`pyvista.UniformGrid`,
the cells are defined as a emergent property based on the shape of the
point array.

To see this in practice, let's create the simplest surface represented
as a :class:`pyvista.PolyData`.  But first, we need to define our points.


Points and Arrays in PyVista
----------------------------
There are a variety of ways to create points within PyVista, and this section shows how to efficiently create an array of points by either:

* Wrapping a VTK array
* Using a :class:`numpy.ndarray` array
* Or just using a :class:`list`

PyVista provides pythonic methods for all three approaches so you can
choose whatever is most efficient for you.  If you're comfortable with
the VTK API, you can choose to wrap VTK arrays, but you may find that
using :class:`numpy.ndarray` is more convienent and avoids the looping
overhead in Python.


Wrapping a VTK Array
~~~~~~~~~~~~~~~~~~~~
Let's define points of a triangle.  Using the VTK API, this can be
done with:

.. jupyter-execute::

   >>> import vtk
   >>> vtk_array = vtk.vtkDoubleArray()
   >>> vtk_array.SetNumberOfComponents(2)
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

However, there's no reason to do this since Python already has the
excellent C array library `NumPy <https://numpy.org/>`_.  You could
more create a points array with:

.. jupyter-execute::

   >>> import numpy as np
   >>> points = np.array([[0, 0, 0],
   ...                    [1, 0, 0],
   ...                    [0.5, 0.667, 0]])
   >>> points

We use a :class:`numpy.ndarray` here so that PyVista directly "point"
the underlying C array to VTK.  VTK already has APIs to directly read
in the C arrays from ``numpy``, and since VTK is written in C++,
everything from Python that is transferred over to VTK needs to be in a
format that VTK can process.

Should you wish to use VTK objects within PyVista, you can still do
this.  In fact, using :func:`pyvista.wrap`, you can even get a numpy-like
representation of the data.  For example:

.. jupyter-execute::

   >>> import pyvista
   >>> wrapped = pyvista.wrap(vtk_array)
   >>> wrapped

Note that when wrapping the underlying VTK array, we actually perform
a shallow copy of the data.  In other words, we pass the pointer from
the underlying C array to the numpy :class:`numpy.ndarray`, meaning
that the two arrays are now efficiently linked.  This means that we
can change the array using numpy array indexing and have it modified
on the "VTK side".

.. jupyter-execute::

   >>> wrapped[0] = 10
   >>> vtk_array.GetValue(0)

If we attempt to pass a Python list, it will be converted over into a
VTK array at some point.

.. jupyter-execute::

   >>> points = [[0, 0, 0],
   ...           [1, 0, 0],
   ...           [0.5, 0.667, 0]]



PLACEHOLDER: Cells
==================

Next, let's define a single cell.  This cell will be composed of three
points in the same order as we defined.

.. note::
   Observe how we had insert a leading ``3`` to tell VTK that our face
   will contain three points.  In our :class:`pyvista.PolyData` VTK
   doesn't assume that faces always contain three points, so we have
   to define that.  This actually gives us the flexibility to define
   as many (or as few as one) points per cell as we wish.

.. jupyter-execute::

   >>> cells = [3, 0, 1, 2]

We now have all the necessary pieces to assemble an instance of
:class:`pyvista.PolyData`.

.. jupyter-execute::

   >>> import pyvista
   >>> mesh = pyvista.PolyData(points, cells)
   >>> mesh
              
This instance has several attributes to access the underlying data of
the mesh.  For example, if you wish to access or modify the points of
the mesh, you can simply:

.. jupyter-execute::

   >>> mesh.points
