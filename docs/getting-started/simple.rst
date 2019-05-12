Basic API Usage
===============

PyVista provides tools to get started with just about any VTK dataset
and wrap that object into an easily accesible data object.
Whether you are new to the VTK library or a power user, the best place to
get started is with PyVista's :func:`pyvista.wrap` and :func:`vtk.read`
functions to either wrap a VTK data object in memory or read a VTK or
VTK-friendly file format.

Wrapping a VTK Data Object
~~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapping function is under the :mod:`pyvista.utilities` module which is
usable from the top level of PyVista:

.. code-block:: python

    import pyvista as pv
    wrapped_data = pv.wrap(my_vtk_data_object)


This allows users to quickly wrap any VTK dataset they have to its appropriate
PyVista object:

.. testcode:: python

    import vtk
    import pyvista as pv
    stuff = vtk.vtkPolyData()
    better = pv.wrap(stuff)


Reading a VTK File
~~~~~~~~~~~~~~~~~~

PyVista provides a convenience function to read VTK file formats into their
respective PyVista data objects. Simply call the :func:`pyvista.read` function
passing the filename:

.. code-block:: python

    import pyvista as pv
    data = pv.read('my_strange_vtk_file.vtk')


Accessing the Wrapped Data Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have a wrapped VTK data object, you can start accessing and
modifying the data! Some of the most common properties to access include the
points and point/cell data (the data attributes assigned to the nodes or cells
of the mesh respectively).

First, check out some common meta data properties:

.. testcode:: python

    import pyvista as pv
    from pyvista import examples
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



Access the points by fetching the ``.points`` attribute on any
PyVista data object:

.. code-block:: python

    >>> the_pts = data.points
    >>> isinstance(the_pts, np.ndarray)
    True

Accessing the different data attributes on the points and cells of the data
object is interfaces via dictionaries with callbacks to the VTK object.
These dictionaries of the different point and cell arrays can be directly
accessed and modified.

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


Plotting
~~~~~~~~

PyVista includes numerous plotting routines that are intended to be intuitive
and highly controllable with ``matplotlib`` similar syntax and keyword
arguments.
To get started, try out the :func:`pyvista.plot` convenience method that is binded
to each PyVista data object:


.. testcode:: python

    import pyvista as pv
    from pyvista import examples

    data = examples.load_airplane()
    data.plot(screenshot='airplane.png')


.. image:: ../images/auto-generated/airplane.png


You can also create the plotter to highly control the scene. First, instantiate
a plotter such as :class:`pyvista.Plotter` or :class:`pyvista.BackgroundPlotter`:

The :class:`pyvista.Plotter` will create a rendering window that will pause the
execution of the code after calling ``show``.

.. testcode:: python

    plotter = pv.Plotter()  # instantiate the plotter
    plotter.add_mesh(data)    # add a dataset to the scene
    cpos = plotter.show()     # show the rendering window


Note that the ``show`` method will return the last used camera position of the
rendering window incase you want to chose a camera position and use it agian
later.

You can then use this cached camera for additional plotting without having to
manually interact with the plotting window:

.. code-block:: python

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(data, color='tan')
    plotter.camera_position = cpos
    plotter.plot(auto_close=False)
    # plotter.screenshot('airplane.png')
    plotter.close()


Be sure to check out all the available plotters for your use case:

* :class:`pyvista.Plotter`: The standard plotter that pauses the code until closed
* :class:`pyvista.BackgroundPlotter`: Creates a rendering window that is interactive and does not pause the code execution
* :class:`pyvista.ScaledPlotter`: An IPython extension of the :class:`pyvista.BackgroundPlotter` that has interactive widgets for scaling the axes in the rendering scene.
