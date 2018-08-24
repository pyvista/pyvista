vtkInterface
============
..
   PyPi
.. image:: https://img.shields.io/pypi/v/vtkInterface.svg

.. image:: https://travis-ci.org/akaszynski/vtkInterface.svg?branch=master
    :target: https://travis-ci.org/akaszynski/vtkInterface

.. image:: https://readthedocs.org/projects/vtkinterface/badge/?version=latest
    :target: https://vtkinterface.readthedocs.io/en/latest/?badge=latest

vtkInterface is a VTK helper module that takes a different approach on interfacing with VTK through numpy and direct array access.  This module simplifies mesh creation and plotting by adding functionality to existing VTK objects.

This module can be used for scientific plotting for presentations and research papers as well as a supporting module for other mesh dependent Python modules.


Documentation
-------------
Refer to the `main documentation page <http://vtkinterface.readthedocs.io/en/latest/index.html>`_ for detailed installation and usage details.

Also see the `wiki <https://github.com/akaszynski/vtkInterface/wiki>`_ for brief code snippets.

Installation
------------
Installation is simply::

    pip install vtkInterface
    
You can also visit `PyPi <http://pypi.python.org/pypi/vtkInterface>`_ or `GitHub <https://github.com/akaszynski/vtkInterface>`_ to download the source.

See the `Installation <http://vtkinterface.readthedocs.io/en/latest/installation.html#install-ref.>`_ for more details if the installation through pip doesn't work out.


Quick Examples
--------------

Loading and Plotting a Mesh from File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loading a mesh is trivial

.. code:: python

    import vtkInterface as vtki
    mesh = vtki.PolyData('airplane.ply')
    mesh.Plot(color='orange')

.. figure:: https://github.com/akaszynski/vtkInterface/raw/master/docs/images/airplane.png
    :width: 500pt

In fact, the code to generate the previous screenshot was created with::

    mesh.Plot(screenshot='airplane.png', color='orange')

The points and faces from the mesh are directly accessible as a numpy array:

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

    >>> print(mesh.GetNumpyFaces())
    [[   0    1    2]
     [   0    2    3]
     [   4    5    1]
     ..., 
     [1324 1333 1323]
     [1325 1216 1334]
     [1325 1334 1324]]
    
    
Creating a Structured Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example creates a simple surface grid and plots the resulting grid and its curvature:

.. code:: python

    import vtkInterface as vtki
    import numpy as np

    # Make data
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    
    # create and plot structured grid
    grid = vtki.StructuredGrid(x, y, z)
    grid.Plot()  # basic plot
    
    # Plot mean curvature
    grid.PlotCurvature()

.. figure:: https://github.com/akaszynski/vtkInterface/raw/master/docs/images/curvature.png
    :width: 500pt


Generating a structured grid is a one liner in this module, and the points from the resulting surface are also a numpy array:

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
This example shows the versatility of the plotting object by generating a moving gif:

.. code:: python
    
    import vtkInterface as vtki
    import numpy as np

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    
    # Create and structured surface
    grid = vtki.StructuredGrid(x, y, z)
    
    # Make copy of points
    pts = grid.points.copy()
    
    # Start a plotter object and set the scalars to the Z height
    plotter = vtki.PlotClass()
    plotter.AddMesh(grid, scalars=z.ravel())
    plotter.Plot(autoclose=False)
    
    # Open a gif
    plotter.OpenGif('wave.gif')
    
    # Update Z and write a frame for each updated position
    nframe = 15
    for phase in np.linspace(0, 2*np.pi, nframe + 1)[:nframe]:
        z = np.sin(r + phase)
        pts[:, -1] = z.ravel()
        plotter.UpdateCoordinates(pts)
        plotter.UpdateScalars(z.ravel())
    
        plotter.WriteFrame()
    
    # Close movie and delete object
    plotter.Close()

.. figure:: https://github.com/akaszynski/vtkInterface/raw/master/docs/images/wave.gif
    :width: 500pt
