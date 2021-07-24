Point-Based Grids
=================

Structured and unstructured grids are designed to manage cells whereas polydata
objects manage surfaces.  The ``pyvista.UnstructuredGrid`` is a derived class
from ``vtk.vtkUnstructuredGrid`` designed to make creation, array access, and
plotting more straightforward than using the VTK object.  The same applies to a
``pyvista.StructuredGrid``.


Unstructured Grid Creation
--------------------------

See :ref:`ref_create_unstructured` for an example on how to create an
unstructured grid from NumPy arrays.


Empty Object
~~~~~~~~~~~~
An unstructured grid can be initialized with:

.. code:: python

    import pyvista as pv
    grid = pv.UnstructuredGrid()

This creates an empty grid, and is not useful until points and cells are added
to it.  VTK points and cells can be added with ``SetPoints`` and ``SetCells``,
but the inputs to these need to be ``vtk.vtkCellArray`` and ``vtk.vtkPoints``
objects, which need to be populated with values.  With PyVista, grid
creation is simplified by initializing the grid directly from numpy
arrays, as demonstrated in the following section.


Loading from File
~~~~~~~~~~~~~~~~~
Unstructured grids can be loaded from a vtk file.

.. jupyter-execute::

    import pyvista as pv
    from pyvista import examples

    grid = pv.UnstructuredGrid(examples.hexbeamfile)
    grid


Structured Grid Creation
------------------------

Empty Object
~~~~~~~~~~~~
A structured grid can be initialized with:

.. code:: python

    import pyvista as pv
    grid = pv.StructuredGrid()

This creates an empty grid, and is not useful until points are added
to it.


Creating from Numpy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~
A structured grid can be created directly from numpy arrays.  This is useful
when creating a grid from scratch or copying it from another format.

Also see :ref:`ref_create_structured` for an example on creating a structured
grid from NumPy arrays.


.. pyvista-plot::
    :context:

    import pyvista as pv
    import numpy as np

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    z = np.arange(-10, 10, 0.25)
    x, y, z = np.meshgrid(x, y, z)

    # create the unstructured grid directly from the numpy arrays and plot
    grid = pv.StructuredGrid(x, y, z)
    grid.plot(show_edges=True)


Loading from File
~~~~~~~~~~~~~~~~~
Structured grids can be loaded from a ``vtk`` file.

.. code:: python

    grid = pv.StructuredGrid(filename)


Plotting Grids
--------------
This example shows how you can load an unstructured grid from a ``vtk`` file and
create a plot and gif movie by updating the plotting object.

.. pyvista-plot::
    :context:

    # Load module and example file
    import pyvista as pv
    from pyvista import examples
    import numpy as np

    # Load example beam grid
    grid = pv.UnstructuredGrid(examples.hexbeamfile)

    # Create fictitious displacements as a function of Z location
    d = np.zeros_like(grid.points)
    d[:, 1] = grid.points[:, 2]**3/250

    # Displace original grid
    grid.points += d

A simple plot can be created with:

.. pyvista-plot::
    :context:

    grid.plot(scalars=d[:, 1], scalar_bar_args={'title': 'Y Displacement'}, cpos='zy', show_edges=True)

A more complex plot can be created with:

.. pyvista-plot::
    :context:

    # Store Camera position.  This can be obtained manually by getting the
    # output of grid.plot()
    # it's hard-coded in this example
    cpos = [(11.915126303095157, 6.11392754955802, 3.6124956735471914),
            (0.0, 0.375, 2.0),
            (-0.42546442225230097, 0.9024244135964158, -0.06789847673314177)]

    # plot this displaced beam
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars=d[:, 1],
                     scalar_bar_args={'title': 'Y Displacement'},
                     rng=[-d.max(), d.max()])
    plotter.add_axes()
    plotter.camera_position = cpos
    plotter.show()


You can animate the motion of the beam by updating the positions and
scalars of the grid copied to the plotting object.


.. pyvista-plot::
    :context:

    plotter = pv.Plotter(window_size=(800, 600))
    plotter.add_mesh(grid, scalars=d[:, 1],
                     show_scalar_bar=False,
                     show_edges=True, rng=[-d.max(), d.max()])
    plotter.add_axes()
    plotter.camera_position = cpos

    # open movie file.  A mp4 file can be written instead.  Requires ``moviepy``
    plotter.open_gif('beam.gif')  # or beam.mp4

    # Modify position of the beam cyclically
    pts = grid.points.copy()  # unmodified points
    for phase in np.linspace(0, 2*np.pi, 20):
        plotter.update_coordinates(pts + d*np.cos(phase))
        plotter.update_scalars(d[:, 1]*np.cos(phase))
        plotter.write_frame()

    # Close the movie and plot
    plotter.close()


You can also render the beam as as a wire-frame object:

.. pyvista-plot::
    :context:

    # Animate plot as a wire-frame
    plotter = pv.Plotter(window_size=(800, 600))
    plotter.add_mesh(grid, scalars=d[:, 1],
                     show_scalar_bar=False,
                     rng=[-d.max(), d.max()], style='wireframe')
    plotter.add_axes()
    plotter.camera_position = cpos

    #plotter.OpenMovie('beam_wireframe.mp4')
    plotter.open_gif('beam_wireframe.gif')
    for phase in np.linspace(0, 2*np.pi, 20):
        plotter.update_coordinates(grid.points + d*np.cos(phase), render=False)
        plotter.update_scalars(d[:, 1]*np.cos(phase), render=False)
        plotter.render()
        plotter.write_frame()

    plotter.close()


Adding Labels to a Plot
-----------------------
Labels can be added to a plot using :func:`add_point_labels()
<pyvista.BasePlotter.add_point_labels>` within the :class:`Plotter <pyvista.BasePlotter>`.
The following example loads the included example beam, generates a
plotting class, and sub-selects points along the y-z plane and labels
their coordinates.  :func:`add_point_labels()
<pyvista.BasePlotter.add_point_labels>` requires that the number of
labels matches the number of points, and that labels is a list
containing one entry per point.  The code automatically converts each
item in the list to a string.

.. pyvista-plot::
    :context:

    # Load module and example file
    import pyvista as pv
    from pyvista import examples

    # Load example beam file
    grid = pv.UnstructuredGrid(examples.hexbeamfile)

    # Create plotting class and add the unstructured grid
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, color='tan')

    # Add labels to points on the yz plane (where x == 0)
    points = grid.points
    mask = points[:, 0] == 0
    plotter.add_point_labels(points[mask], points[mask].tolist())

    plotter.camera_position = [
                    (-1.4643015810492384, 1.5603923627830638, 3.16318236536270),
                    (0.05268120500967251, 0.639442034364944, 1.204095304165153),
                    (0.2364061044392675, 0.9369426029156169, -0.25739213784721)]

    plotter.show()


This example is similar and shows how labels can be combined with a
scalar bar to show the exact value of certain points.

.. pyvista-plot::
    :context:

    # Label the Z position
    values = grid.points[:, 2]

    # Create plotting class and add the unstructured grid
    plotter = pv.Plotter()
    # color mesh according to z value
    plotter.add_mesh(grid, scalars=values,
                     scalar_bar_args={'title': 'Z Position'},
                     show_edges=True)

    # Add labels to points on the yz plane (where x == 0)
    mask = grid.points[:, 0] == 0
    plotter.add_point_labels(points[mask], values[mask].tolist(), font_size=24)

    # add some text to the plot
    plotter.add_text('Example showing plot labels')

    plotter.view_vector((-6, -3, -4), (0.,-1., 0.))
    plotter.show()


pv.UnstructuredGrid Class Methods
---------------------------------
The following is a description of the methods available to a
``pv.UnstructuredGrid`` object.  It inherits all methods from the original
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


Explicit Structured Grid
------------------------

.. rubric:: Attributes
.. autoautosummary:: pyvista.ExplicitStructuredGrid
   :attributes:

.. rubric:: Methods
.. autoautosummary:: pyvista.ExplicitStructuredGrid
   :methods:

.. autoclass:: pyvista.ExplicitStructuredGrid
   :show-inheritance:
   :members:
   :undoc-members:


pv.StructuredGrid Class Methods
-------------------------------
The following is a description of the methods available to a
``pv.StructuredGrid`` object.  It inherits all methods from the original
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
These methods are common to both ``pv.StructuredGrid`` and
``pv.UnstructuredGrid`` objects.



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
