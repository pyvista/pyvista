Structured and Unstructured Grids
=================================
Structured and unstructured grids are designed to manage cells whereas a polydata object manage surfaces.  The ``vtk.UnstructuredGrid`` is derived class from ``vtk.vtkUnstructuredGrid`` designed to make creation, array access, and plotting more straightforward than using the vtk object.  The same goes with a ``vtk.StructuredGrid``.


Unstructured Grid Creation
--------------------------

Empty Object
~~~~~~~~~~~~
An unstructured grid can be initialized with:

.. code:: python

    import vtkInterface
    grid = vtkInterface.UnstructuredGrid()

This creates an empty grid, and is not useful until points and cells are added to it.  VTK points and cells can be added with ``SetPoints`` and ``SetCells``, but the inputs to these need to be ``vtk.vtkCellArray`` and ``vtk.vtkPoints`` objects, which need to be populated with values.  Grid creation is simplified by initializing the grid directly from numpy arrays as in the following section.


Creating from Numpy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~
An unstructured grid can be created directly from numpy arrays.  This is useful when creating a grid from scratch or copying it from another format.  See `vtkUnstructuredGrid <https://www.vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html>`_ for available cell types and their descriptions.

.. code:: python

    import vtkInterface

    # offset array.  Identifies the start of each cell in the cells array
    offset = np.array([0, 9])

    # Contains information on the points composing each cell.
    # Each cell begins with the number of points in the cell and then the points
    # composing the cell
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])

    # cell type array. Contains the cell type of each cell
    cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON], np.int8)

    cell1 = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [1, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [1, 1, 1],
                      [0, 1, 1]])

    cell2 = np.array([[0, 0, 2],
                      [1, 0, 2],
                      [1, 1, 2],
                      [0, 1, 2],
                      [0, 0, 3],
                      [1, 0, 3],
                      [1, 1, 3],
                      [0, 1, 3]])

    # points of the cell array
    points = np.vstack((cell1, cell2))

    # create the unstructured grid directly from the numpy arrays
    grid = vtkInterface.UnstructuredGrid(offset, cells, cell_type, points)

    # plot the grid
    grid.Plot()

..
   The resulting plot can be found in :numref:`twocubes`.

.. image:: ./images/twocubes.png

Loading from File
~~~~~~~~~~~~~~~~~
Unstructured grids can be loaded from a vtk file.

.. code:: python

    grid = vtkInterface.UnstructuredGrid(filename)


Structured Grid Creation
------------------------

Empty Object
~~~~~~~~~~~~
A structured grid can be initialized with:

.. code:: python

    import vtkInterface
    grid = vtkInterface.StructuredGrid()

This creates an empty grid, and is not useful until points are added to it and the shape set using ``SetPoints`` and ``SetDimensions``.  This can be done with:


.. code:: python

    import numpy as np
    import vtkInterface

    # create a cube of points
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    z = np.arange(-10, 10, 0.25)
    x, y, z = np.meshgrid(x, y, z)

    # convert 
    points = np.empty((x.size, 3))
    points[:, 0] = x.ravel('F')
    points[:, 1] = y.ravel('F')
    points[:, 2] = z.ravel('F')

    # Create structured grid
    grid = vtkInterface.StructuredGrid()
    grid.SetDimensions(x.shape)
    grid.SetPoints(vtkInterface.MakevtkPoints(points))


Creating from Numpy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~
A structured grid can be created directly from numpy arrays.  This is useful when creating a grid from scratch or copying it from another format.

.. code:: python

    import vtkInterface

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    z = np.arange(-10, 10, 0.25)
    x, y, z = np.meshgrid(x, y, z)

    # create the unstructured grid directly from the numpy arrays and plot
    grid = vtkInterface.StructuredGrid(x, y, z)
    grid.Plot()

.. image:: ./images/structured_cube.png


Loading from File
~~~~~~~~~~~~~~~~~
Structured grids can be loaded from a vtk file.

.. code:: python

    grid = vtkInterface.StructuredGrid(filename)


Plotting Grids
--------------
This example shows how you can load an unstructured grid from a vtk file and create a plot and gif movie by updating the plotting object.

.. code:: python

    # Load module and example file
    import vtkInterface
    from vtkInterface import examples
    import numpy as np
    
    # Load example beam grid
    grid = vtkInterface.UnstructuredGrid(examples.hexbeamfile)
    
    # Create fictitious displacements as a function of Z location
    d = np.zeros_like(grid.points)
    d[:, 1] = grid.points[:, 2]**3/250
    
    # Displace original grid
    grid.points += d

A simple plot can be created by using:

.. code:: python

    grid.Plot(scalars=d[:, 1], stitle='Y Displacement')

A more complex plot can be created using:

.. code:: python

    # Store Camera position.  This can be obtained manually by getting the
    # output of grid.Plot()
    # it's hard-coded in this example
    cpos = [(11.915126303095157, 6.11392754955802, 3.6124956735471914),
            (0.0, 0.375, 2.0),
            (-0.42546442225230097, 0.9024244135964158, -0.06789847673314177)]
    
    # plot this displaced beam
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(grid, scalars=d[:, 1], stitle='Y Displacement', 
                  rng=[-d.max(), d.max()])
    plobj.AddAxes()
    plobj.SetCameraPosition(cpos)
    
    # Don't let it close automatically so we can take a screen-shot
    cpos = plobj.Plot(autoclose=False)
    plobj.TakeScreenShot('beam.png')
    plobj.Close()

.. image:: ./images/beam.png

You can animate the motion of the beam by updating the positions and scalars of the grid copied to the plotting object.  First you have to setup the plotting object:

.. code:: python

    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(grid, scalars=d[:, 1], stitle='Y Displacement', 
                  showedges=True, rng=[-d.max(), d.max()], 
                  interpolatebeforemap=True)
    plobj.AddAxes()
    plobj.SetCameraPosition(cpos)
    
You then open the render window by plotting before opening movie file.  Set autoclose to False so the plobj does not close automatically.  Disabling interactive means the plot will automatically continue without waiting for the user to exit the window.

.. code:: python

    plobj.Plot(interactive=False, autoclose=False, window_size=[800, 600])
    
    # open movie file.  A mp4 file can be written instead.  Requires moviepy
    plobj.OpenGif('beam.gif')  # or beam.mp4
    
    # Modify position of the beam cyclically
    pts = grid.points.copy()  # unmodified points
    for phase in np.linspace(0, 2*np.pi, 20):
        plobj.UpdateCoordinates(pts + d*np.cos(phase), render=False)
        plobj.UpdateScalars(d[:, 1]*np.cos(phase), render=False)
        plobj.Render()
        plobj.WriteFrame()
    
    # Close the movie and plot
    plobj.Close()
    
.. image:: ./images/beam.gif

You can also render the beam as as a wire-frame object:

.. code:: python

    # Animate plot as a wire-frame
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(grid, scalars=d[:, 1], stitle='Y Displacement', showedges=True,
                  rng=[-d.max(), d.max()], interpolatebeforemap=True,
                  style='wireframe')
    plobj.AddAxes()
    plobj.SetCameraPosition(cpos)
    plobj.Plot(interactive=False, autoclose=False, window_size=[800, 600])
    
    #plobj.OpenMovie('beam.mp4')
    plobj.OpenGif('beam_wireframe.gif')
    for phase in np.linspace(0, 2*np.pi, 20):
        plobj.UpdateCoordinates(pts + d*np.cos(phase), render=False)
        plobj.UpdateScalars(d[:, 1]*np.cos(phase), render=False)
        plobj.Render()
        plobj.WriteFrame()
    
    plobj.Close()
    
.. image:: ./images/beam_wireframe.gif


Adding Labels to a Plot
-----------------------
Labels can be added to a plot using the ``AddPointLabels`` function within the ``PlotClass`` object.  The following example loads the included example beam, generates a plotting class, and sub-selects points along the y-z plane and labels their coordinates.  ``AddPointLabels`` requires that the number of labels matches the number of points, and that labels is a list containing one entry per point.  The code automatically converts each item in the list to a string.

.. code:: python

    # Load module and example file
    import vtkInterface
    from vtkInterface import examples

    # Load example beam file
    grid = vtkInterface.UnstructuredGrid(examples.hexbeamfile)

    # Create plotting class and add the unstructured grid
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(grid)

    # Add labels to points on the yz plane (where x == 0)
    points = grid.GetNumpyPoints()
    mask = points[:, 0] == 0
    plobj.AddPointLabels(points[mask], points[mask].tolist())

    plobj.Plot()

.. image:: ./images/labels0.png

This example is similar and shows how labels can be combined with a scalar bar to show the exact value of certain points.

.. code:: python

    # Label the Z position
    values = grid.points[:, 2]

    # Create plotting class and add the unstructured grid
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(grid, scalars=values) # color mesh according to z value
    plobj.AddScalarBar(title='Z Position')

    # Add labels to points on the yz plane (where x == 0)
    mask = grid.points[:, 0] == 0
    plobj.AddPointLabels(points[mask], values[mask].tolist(), fontsize=24)

    # add some text to the plot
    plobj.AddText('Example showing plot labels')

    plobj.Plot()

.. image:: ./images/labels1.png




vtkInterface.Unstructured Grid Class Methods
--------------------------------------------
The following is a description of the methods available to a ``vtkInterface.UnstructuredGrid`` object.  It inherits all methods from the original vtk object, `vtk.vtkUnstructuredGrid <https://www.vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html>`_.

.. autoclass:: vtkInterface.UnstructuredGrid
    :members:


vtkInterface.Structured Grid Class Methods
--------------------------------------------
The following is a description of the methods available to a ``vtkInterface.StructuredGrid`` object.  It inherits all methods from the original vtk object, `vtk.vtkStructuredGrid <https://www.vtk.org/doc/nightly/html/classvtkStructuredGrid.html>`_.

.. autoclass:: vtkInterface.StructuredGrid
    :members:


Methods in Common with Structured and Unstructured Grids
--------------------------------------------------------
These methods are in common to both ``vtkInterface.StructuredGrid`` and ``vtkInterface.UnstructuredGrid`` objects.

.. autoclass:: vtkInterface.Grid
    :members:
