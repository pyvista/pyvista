vtki
====

.. image:: https://img.shields.io/pypi/v/vtki.svg
   :target: https://pypi.org/project/vtki/

.. image:: https://travis-ci.org/akaszynski/vtki.svg?branch=master
   :target: https://travis-ci.org/akaszynski/vtki

.. image:: https://readthedocs.org/projects/vtkinterface/badge/?version=latest
   :target: https://vtkinterface.readthedocs.io/en/latest/?badge=latest

``vtki`` is a VTK helper module that takes a different approach on interfacing
with VTK through numpy and direct array access.  This module simplifies mesh
creation and plotting by adding functionality to existing VTK objects.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.


Documentation
-------------
Refer to the detailed `readthedocs <http://vtkInterface.readthedocs.io/en/latest/index.html>`_
documentation for detailed installation and usage details.

Also see the `wiki <https://github.com/akaszynski/vtki/wiki>`_ for brief code
snippets.

Installation
------------
Installation is simply::

    pip install vtki

You can also visit `PyPi <http://pypi.python.org/pypi/vtki>`_ or
`GitHub <https://github.com/akaszynski/vtki>`_ to download the source.

See the `Installation <http://vtkInterface.readthedocs.io/en/latest/installation.html#install-ref.>`_
for more details if the installation through pip doesn't work out.


Quick Examples
--------------

Loading and Plotting a Mesh from File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loading a mesh is trivial

.. code:: python

    import vtki
    mesh = vtki.PolyData('airplane.ply')
    mesh.plot(color='orange')

.. figure:: https://github.com/akaszynski/vtki/raw/master/docs/images/airplane.png
    :width: 500pt

In fact, the code to generate the previous screenshot was created in one line with::

    mesh.plot(screenshot='airplane.png', color='orange')

The points and faces from the mesh are directly accessible as a NumPy array:

.. code:: python

    >>> print(mesh.points)
    [[ 896.99401855   48.76010132   82.26560211]
     [ 906.59301758   48.76010132   80.74520111]
     [ 907.53900146   55.49020004   83.65809631]
     ...,
     [ 806.66497803  627.36297607    5.11482   ]
     [ 806.66497803  654.43200684    7.51997995]
     [ 806.66497803  681.5369873     9.48744011]]

.. code:: python

    >>> faces = mesh.faces.reshape(-1, 4)
    >>> print(faces[:, 1:])
    [[   0    1    2]
     [   0    2    3]
     [   4    5    1]
     ...,
     [1324 1333 1323]
     [1325 1216 1334]
     [1325 1334 1324]]


Creating a Structured Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example creates a simple surface grid and plots the resulting grid and its
curvature:

.. code:: python

    import vtki
    import numpy as np

    # Make data
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)

    # create and plot structured grid
    grid = vtki.StructuredGrid(x, y, z)
    grid.plot()  # basic plot

    # Plot mean curvature
    grid.plot_curvature()

.. figure:: https://github.com/akaszynski/vtki/raw/master/docs/images/curvature.png
    :width: 500pt


Generating a structured grid is a one liner in this module, and the points from
the resulting surface are also a NumPy array:

.. code:: python

    >>> grid.points
    [[-10.         -10.           0.99998766]
     [ -9.75       -10.           0.98546793]
     [ -9.5        -10.           0.9413954 ]
     ...,
     [  9.25         9.75         0.76645876]
     [  9.5          9.75         0.86571785]
     [  9.75         9.75         0.93985707]]


Creating a GIF Movie
~~~~~~~~~~~~~~~~~~~~
This example shows the versatility of the plotting object by generating a moving
gif:

.. code:: python

    import vtki
    import numpy as np

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)

    # Create and structured surface
    grid = vtki.StructuredGrid(x, y, z)

    # Creat a plotter object and set the scalars to the Z height
    plotter = vtki.Plotter()
    plotter.add_mesh(grid, scalars=z.ravel())

    # setup camera and close
    plotter.plot(autoclose=False)

    # Open a gif
    plotter.open_gif('wave.gif')

    pts = grid.points.copy()

    # Update Z and write a frame for each updated position
    nframe = 15
    for phase in np.linspace(0, 2*np.pi, nframe + 1)[:nframe]:
        z = np.sin(r + phase)
        pts[:, -1] = z.ravel()
        plotter.update_coordinates(pts)
        plotter.update_scalars(z.ravel())
        plotter.write_frame()

    # Close movie and delete object
    plotter.close()


.. figure:: https://github.com/akaszynski/vtki/raw/master/docs/images/wave.gif
   :width: 500pt


Using Common Filters
--------------------

``vtki`` wrapped data objects have a suite of common filters ready for immediate
use directly on the object. These filters include:

* ``slice``: creates a single slice through the input dataset on a user defined plane
* ``slice_orthogonal``: creates a ``MultiBlock`` dataset of three orthogonal slices
* ``slice_along_axis``: creates a ``MultiBlock`` dataset of many slices along a specified axis
* ``threshold``: Thresholds a dataset by a single value or range of values
* ``threshold_percent``: Threshold by percentages of the scalar range
* ``clip``: Clips the dataset by a user defined plane
* ``outline_corners``: Outlines the corners of the data extent
* ``extract_geometry``: Extract surface geometry


To use these filter, call the method of your choice directly on your data object:


.. code:: python

    from vtki import examples
    dataset = examples.load_uniform()

    # Apply a threshold over a data range
    result = dataset.threshold([300, 500])



iPython Interactive Plotting Tools
----------------------------------

``vtki`` comes packed with several interactive plotting tools to make using the
filters a bit more intuitive. If in an iPython environment, call one of the
tools on an input dataset to yield widgets that will control a filter or task in
an interactive rendering scene:

.. code:: python

    import vtki
    from vtki import examples
    dataset = examples.load_uniform()

    # Use the slicer tool
    vtki.OthogonalSlicer(dataset)


.. figure:: https://github.com/akaszynski/vtki/raw/master/docs/images/slicer-tool.gif
   :width: 500pt
