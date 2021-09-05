.. _point_sets_api:

Point Sets
==========

PyVista point sets are datasets with explicit geometry where the point
and cell topology are specified and not inferred.  PyVista point sets
are modeled after the same data model as VTK's point sets while being
designed to make creation, array access, and plotting more
straightforward than their VTK counterparts.

The :class:`pyvista.UnstructuredGrid` class is used for arbitrary
combinations of all possible cell types:

.. pyvista-plot::
   :include-source: False

   from pyvista import demos
   demos.plot_datasets('UnstructuredGrid')


The :class:`pyvista.PolyData` is used for datasets consisting of surface
geometry (e.g. vertices, lines, and polygons):

.. pyvista-plot::
   :include-source: False

   from pyvista import demos
   demos.plot_datasets('PolyData')


The :class:`pyvista.StructuredGrid` is used for topologically regular arrays of
data:

.. pyvista-plot::
   :include-source: False

   from pyvista import demos
   demos.plot_datasets('StructuredGrid')


**Class Descriptions**

The following table describes PyVista's point set classes.  These
classes inherit all methods from their corresponding VTK
`vtkPolyData`_, `vtkUnstructuredGrid`_, `vtkStructuredGrid`_, and
`vtkExplicitStructuredGrid`_ superclasses.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   pyvista.PolyData
   pyvista.UnstructuredGrid
   pyvista.StructuredGrid
   pyvista.ExplicitStructuredGrid

.. _vtkPolyData: https://www.vtk.org/doc/nightly/html/classvtkPolyData.html
.. _vtkUnstructuredGrid: https://www.vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html
.. _vtkStructuredGrid: https://www.vtk.org/doc/nightly/html/classvtkStructuredGrid.html
.. _vtkExplicitStructuredGrid: https://vtk.org/doc/nightly/html/classvtkExplicitStructuredGrid.html


PolyData Creation
-----------------

See :ref:`ref_create_poly` for an example on creating a
:class:`pyvista.PolyData` object from NumPy arrays.


Empty Object
~~~~~~~~~~~~
A polydata object can be initialized with:

.. jupyter-execute::

    import pyvista
    grid = pyvista.PolyData()

This creates an empty grid, and is not useful until points and cells
are added to it.  VTK points and cells can be added with ``SetPoints``
and ``SetCells``, but the inputs to these need to be
``vtk.vtkCellArray`` and ``vtk.vtkPoints`` objects, which need to be
populated with values.  Grid creation is simplified by initializing
the grid directly from NumPy arrays as in the following section.


Initialize from a File
~~~~~~~~~~~~~~~~~~~~~~
Both binary and ASCII .ply, .stl, and .vtk files can be read using
PyVista.  For example, the PyVista package contains example meshes and
these can be loaded with:

.. jupyter-execute::

    import pyvista
    from pyvista import examples

    # Load mesh
    mesh = pyvista.PolyData(examples.planefile)
    mesh

This mesh can then be written to a .vtk file using:

.. code:: python

    mesh.save('plane.vtk')

These meshes are identical.

.. code:: python

    import numpy as np

    mesh_from_vtk = pyvista.PolyData('plane.vtk')
    print(np.allclose(mesh_from_vtk.points, mesh.points))


Mesh Manipulation and Plotting
------------------------------
Meshes can be directly manipulated using NumPy or with the built-in
translation and rotation routines.  This example loads two meshes and
moves, scales, copies them, and finally plots them.

To plot more than one mesh a :class:`pyvista.Plotter` instance must be
created to manage the plotting.  The following code creates a plotter
and plots the meshes with various colors.


.. pyvista-plot::
    :context:

    import pyvista
    from pyvista import examples

    # load and shrink airplane
    airplane = pyvista.PolyData(examples.planefile)
    airplane.points /= 10 # shrink by 10x

    # rotate and translate ant so it is on the plane
    ant = pyvista.PolyData(examples.antfile)
    ant.rotate_x(90)
    ant.translate([90, 60, 15])

    # Make a copy and add another ant
    ant_copy = ant.copy()
    ant_copy.translate([30, 0, -10])

    # Create plotter object
    plotter = pyvista.Plotter()
    plotter.add_mesh(ant, 'r')
    plotter.add_mesh(ant_copy, 'b')

    # Add airplane mesh and make the color equal to the Y position.  Add a
    # scalar bar associated with this mesh
    plane_scalars = airplane.points[:, 1]
    plotter.add_mesh(airplane, scalars=plane_scalars,
                     scalar_bar_args={'title': 'Airplane Y\nLocation'})

    # Add annotation text
    plotter.add_text('Ants and Plane Example')
    plotter.show()

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
    # output of grid.plot
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
