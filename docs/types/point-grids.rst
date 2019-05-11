Point-Based Grids
=================

Structured and unstructured grids are designed to manage cells whereas a
polydata object manage surfaces.  The ``vtk.UnstructuredGrid`` is derived class
from ``vtk.vtkUnstructuredGrid`` designed to make creation, array access, and
plotting more straightforward than using the vtk object.  The same goes with a
``vtk.StructuredGrid``.


Unstructured Grid Creation
--------------------------

See :ref:`ref_create_unstructured` for an example on how to create an
unstructured grid from NumPy arrays.

Empty Object
~~~~~~~~~~~~
An unstructured grid can be initialized with:

.. testcode:: python

    import pyvista
    grid = pyvista.UnstructuredGrid()

This creates an empty grid, and is not useful until points and cells are added
to it.  VTK points and cells can be added with ``SetPoints`` and ``SetCells``,
but the inputs to these need to be ``vtk.vtkCellArray`` and ``vtk.vtkPoints``
objects, which need to be populated with values.  Grid creation is simplified
by initializing the grid directly from numpy arrays as in the following section.


Loading from File
~~~~~~~~~~~~~~~~~
Unstructured grids can be loaded from a vtk file.

.. testcode:: python

    import pyvista
    from pyvista import examples

    grid = pyvista.UnstructuredGrid(examples.hexbeamfile)


Structured Grid Creation
------------------------

Empty Object
~~~~~~~~~~~~
A structured grid can be initialized with:

.. testcode:: python

    import pyvista
    grid = pyvista.StructuredGrid()

This creates an empty grid, and is not useful until points are added
to it.


Creating from Numpy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~
A structured grid can be created directly from numpy arrays.  This is useful
when creating a grid from scratch or copying it from another format.

Also see :ref:`ref_create_structured` for an example on creating a structured
grid from NumPy arrays.

.. testcode:: python

    import pyvista
    import numpy as np

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    z = np.arange(-10, 10, 0.25)
    x, y, z = np.meshgrid(x, y, z)

    # create the unstructured grid directly from the numpy arrays and plot
    grid = pyvista.StructuredGrid(x, y, z)
    grid.plot(show_edges=True, screenshot='structured_cube.png')

.. image:: ../images/auto-generated/structured_cube.png


Loading from File
~~~~~~~~~~~~~~~~~
Structured grids can be loaded from a ``vtk`` file.

.. code:: python

    grid = pyvista.StructuredGrid(filename)


Plotting Grids
--------------
This example shows how you can load an unstructured grid from a ``vtk`` file and
create a plot and gif movie by updating the plotting object.

.. testcode:: python

    # Load module and example file
    import pyvista
    from pyvista import examples
    import numpy as np

    # Load example beam grid
    grid = pyvista.UnstructuredGrid(examples.hexbeamfile)

    # Create fictitious displacements as a function of Z location
    d = np.zeros_like(grid.points)
    d[:, 1] = grid.points[:, 2]**3/250

    # Displace original grid
    grid.points += d

A simple plot can be created by using:

.. testcode:: python

    grid.plot(scalars=d[:, 1], stitle='Y Displacement')

A more complex plot can be created using:

.. testcode:: python

    # Store Camera position.  This can be obtained manually by getting the
    # output of grid.plot()
    # it's hard-coded in this example
    cpos = [(11.915126303095157, 6.11392754955802, 3.6124956735471914),
            (0.0, 0.375, 2.0),
            (-0.42546442225230097, 0.9024244135964158, -0.06789847673314177)]

    # plot this displaced beam
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars=d[:, 1], stitle='Y Displacement',
                  rng=[-d.max(), d.max()])
    plotter.add_axes()
    plotter.camera_position = cpos

    # Don't let it close automatically so we can take a screenshot
    cpos = plotter.plot(auto_close=False)
    plotter.screenshot('beam.png')
    plotter.close()

.. image:: ../images/auto-generated/beam.png

You can animate the motion of the beam by updating the positions and scalars of
the grid copied to the plotting object.
First you have to setup the plotting object:

.. testcode:: python

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars=d[:, 1], stitle='Y Displacement',
                  show_edges=True, rng=[-d.max(), d.max()],
                  interpolate_before_map=True)
    plotter.add_axes()
    plotter.camera_position = cpos

You then open the render window by plotting before opening movie file.
Set auto_close to False so the plotter does not close automatically.
Disabling interactive means the plot will automatically continue without waiting
for the user to exit the window.

.. testcode:: python

    plotter.plot(interactive=False, auto_close=False, window_size=[800, 600])

    # open movie file.  A mp4 file can be written instead.  Requires moviepy
    plotter.open_gif('beam.gif')  # or beam.mp4

    # Modify position of the beam cyclically
    pts = grid.points.copy()  # unmodified points
    for phase in np.linspace(0, 2*np.pi, 20):
        plotter.update_coordinates(pts + d*np.cos(phase))
        plotter.update_scalars(d[:, 1]*np.cos(phase))
        plotter.write_frame()

    # Close the movie and plot
    plotter.close()

.. image:: ../images/auto-generated/beam.gif

You can also render the beam as as a wire-frame object:

.. testcode:: python

    # Animate plot as a wire-frame
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars=d[:, 1], stitle='Y Displacement', show_edges=True,
                  rng=[-d.max(), d.max()], interpolate_before_map=True,
                  style='wireframe')
    plotter.add_axes()
    plotter.camera_position = cpos
    plotter.show(interactive=False, auto_close=False, window_size=[800, 600])

    #plotter.OpenMovie('beam.mp4')
    plotter.open_gif('beam_wireframe.gif')
    for phase in np.linspace(0, 2*np.pi, 20):
        plotter.update_coordinates(grid.points + d*np.cos(phase), render=False)
        plotter.update_scalars(d[:, 1]*np.cos(phase), render=False)
        plotter.render()
        plotter.write_frame()

    plotter.close()

.. image:: ../images/auto-generated/beam_wireframe.gif


Adding Labels to a Plot
-----------------------
Labels can be added to a plot using the ``add_point_labels`` function within the
``Plotter`` object.  The following example loads the included example beam,
generates a plotting class, and sub-selects points along the y-z plane and
labels their coordinates.  ``add_point_labels`` requires that the number of
labels matches the number of points, and that labels is a list containing one
entry per point.  The code automatically converts each item in the list to a
string.

.. testcode:: python

    # Load module and example file
    import pyvista
    from pyvista import examples

    # Load example beam file
    grid = pyvista.UnstructuredGrid(examples.hexbeamfile)

    # Create plotting class and add the unstructured grid
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, color='tan')

    # Add labels to points on the yz plane (where x == 0)
    points = grid.points
    mask = points[:, 0] == 0
    plotter.add_point_labels(points[mask], points[mask].tolist())

    plotter.camera_position = [
                    (-1.4643015810492384, 1.5603923627830638, 3.16318236536270),
                    (0.05268120500967251, 0.639442034364944, 1.204095304165153),
                    (0.2364061044392675, 0.9369426029156169, -0.25739213784721)]

    plotter.show(screenshot='labels0.png')

.. image:: ../images/auto-generated/labels0.png

This example is similar and shows how labels can be combined with a scalar bar
to show the exact value of certain points.

.. testcode:: python

    # Label the Z position
    values = grid.points[:, 2]

    # Create plotting class and add the unstructured grid
    plotter = pyvista.Plotter(notebook=False)
    # color mesh according to z value
    plotter.add_mesh(grid, scalars=values, stitle='Z Position', show_edges=True)

    # Add labels to points on the yz plane (where x == 0)
    mask = grid.points[:, 0] == 0
    plotter.add_point_labels(points[mask], values[mask].tolist(), font_size=24)

    # add some text to the plot
    plotter.add_text('Example showing plot labels')

    plotter.view_vector((-6, -3, -4), (0.,-1., 0.))
    plotter.show(screenshot='labels1.png')

.. image:: ../images/auto-generated/labels1.png




pyvista.Unstructured Grid Class Methods
--------------------------------------------
The following is a description of the methods available to a
``pyvista.UnstructuredGrid`` object.  It inherits all methods from the original
``vtk`` object, `vtk.vtkUnstructuredGrid <https://www.vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html>`_.



.. rubric:: Attributes

.. autoautosummary:: pyvista.UnstructuredGrid
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.UnstructuredGrid
   :methods:


.. autoclass:: pyvista.UnstructuredGrid
   :show-inheritance:
   :members:
   :undoc-members:


pyvista.Structured Grid Class Methods
--------------------------------------------
The following is a description of the methods available to a
``pyvista.StructuredGrid`` object.  It inherits all methods from the original
``vtk`` object, `vtk.vtkStructuredGrid <https://www.vtk.org/doc/nightly/html/classvtkStructuredGrid.html>`_.



.. rubric:: Attributes

.. autoautosummary:: pyvista.StructuredGrid
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.StructuredGrid
   :methods:

.. autoclass:: pyvista.StructuredGrid
   :show-inheritance:
   :members:
   :undoc-members:


Methods in Common with Structured and Unstructured Grids
--------------------------------------------------------
These methods are in common to both ``pyvista.StructuredGrid`` and
``pyvista.UnstructuredGrid`` objects.



.. rubric:: Attributes

.. autoautosummary:: pyvista.PointGrid
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.PointGrid
   :methods:

.. autoclass:: pyvista.PointGrid
   :show-inheritance:
   :members:
   :undoc-members:
