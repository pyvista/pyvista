Basic API Usage
===============

``vtki`` provides tools to easily get started with just about any VTK dataset
and wrap access to that object into an easily accesible data object.
Whether your new to the VTK library or a power user, the best place to get
started is with ``vtki``'s :func:`vtki.wrap` and :func:`vtk.read`
functions to either wrap a VTK data object in memory or read a VTK or
VTK-friendly file format.

Wrapping a VTK Data Object
~~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapping function is under the :mod:`vtki.utilities` module which is
usable from the top level of ``vtki``:

.. code-block:: python

    import vtki
    wrapped_data = vtki.wrap(my_vtk_data_object)


This allows users to quickly wrap any VTK dataset they have to its appropriate
`vtki` object:

.. testcode:: python

    import vtk
    import vtki
    stuff = vtk.vtkPolyData()
    better = vtki.wrap(stuff)


Reading a VTK File
~~~~~~~~~~~~~~~~~~

``vtki`` provides a convenience function to read VTK file formats into their
respective ``vtki`` data objects. Simply call the :func:`vtki.read` function
passing the filename:

.. code-block:: python

    import vtki
    data = vtki.read('my_strange_vtk_file.vtk')


Accessing the Wrapped Data Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have a wrapped VTK data object, you can start accessing and
modifying the data! Some of the most common properties to access include the
points and point/cell data (the data attribute assigned to the nodes or faces
of the mesh respectively).

First, lets check out some common meta data properties:

.. code-block:: python

    import vtki
    from vtki import examples
    import numpy as np

.. code-block:: python

    >>> data = examples.load_airplane()
    >>> # Inspect how many cells are in this dataset
    >>> data.n_cells
    2452
    >>> # Inspect how many points are in this dataset
    >>> data.n_points
    1335
    >>> # What about scalar arrays? Are there any?
    >>> data.n_scalars
    0
    >>> # What are the data bounds?
    >>> data.bounds
    [139.06100463867188, 1654.9300537109375, 32.09429931640625, 1319.949951171875, -17.741199493408203, 282.1300048828125]
    >>> # Hm, where is the center of this dataset?
    >>> data.center
    [896.9955291748047, 676.0221252441406, 132.19440269470215]



Accessing the points is easy! Simply call the ``.points`` attribute on any
``vtki`` data object:

.. code-block:: python

    >>> the_pts = data.points
    >>> isinstance(the_pts, np.ndarray)
    True

Accessing the different data attributes on the points and cells of the data
object is also easy! The ``vtki`` data objects have a dictionary of the
different point and cell arrays that you can directly access and modify.

.. code-block:: python

    >>> data = examples.load_uniform()
    >>> # Fetch a data array from the point data
    >>> arr = data.point_arrays['Spatial Point Data']
    >>> # Assign a new array to the cell data:
    >>> data.cell_arrays['foo'] = np.random.rand(data.n_cells)
    >>> # Don't remember if your array is point or cell data? Doesn't matter!
    >>> foo = data.get_scalar('foo')
    >>> isinstance(foo, np.ndarray)
    True
