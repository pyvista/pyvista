Point Data
==========
The ``pyvista.PolyData`` object adds additional functionality to the
``vtk.vtkPolyData`` object, to include direct array access through NumPy,
one line plotting, and other mesh functions.


PolyData Creation
-----------------

See :ref:`ref_create_poly` for an example on creating a ``pyvista.PolyData`` object
from NumPy arrays.

Empty Object
~~~~~~~~~~~~
A polydata object can be initialized with:

.. jupyter-execute::

    import pyvista
    grid = pyvista.PolyData()

This creates an empty grid, and is not useful until points and cells are added
to it.  VTK points and cells can be added with ``SetPoints`` and ``SetCells``,
but the inputs to these need to be ``vtk.vtkCellArray`` and ``vtk.vtkPoints``
objects, which need to be populated with values.
Grid creation is simplified by initializing the grid directly from NumPy
arrays as in the following section.


Initialize from a File
~~~~~~~~~~~~~~~~~~~~~~
Both binary and ASCII .ply, .stl, and .vtk files can be read using PyVista.
For example, the PyVista package contains example meshes and these can be loaded with:

.. jupyter-execute::
    :hide-code:

    import pyvista
    pyvista.set_jupyter_backend('panel')
    pyvista.set_plot_theme('document')


.. jupyter-execute::

    import pyvista
    from pyvista import examples

    # Load mesh
    mesh = pyvista.PolyData(examples.planefile)
    mesh

This mesh can then be written to a vtk file using:

.. jupyter-execute::

    mesh.save('plane.vtk')

These meshes are identical.

.. code:: python

    import numpy as np

    mesh_from_vtk = pyvista.PolyData('plane.vtk')
    print(np.allclose(mesh_from_vtk.points, mesh.points))


.. jupyter-execute::
    :hide-code:

    import os
    try:
        os.remove('plane.vtk')
    except FileNotFoundError:
        pass


Mesh Manipulation and Plotting
------------------------------
Meshes can be directly manipulated using NumPy or with the built-in
translation and rotation routines.  This example loads two meshes and
moves, scales, copies them, and finally plots them.

To plot more than one mesh a plotting class must be created to manage
the plotting.  The following code creates the class and plots the
meshes with various colors.

.. jupyter-execute::
   :hide-code:

   # must have this here as our global backend may not be static
   import pyvista
   pyvista.set_jupyter_backend('static')


.. jupyter-execute::

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

    # Create plotting object
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


pyvista.PolyData Grid Class Methods
----------------------------------------
The following is a description of the methods available to a ``pyvista.PolyData``
object.  It inherits all methods from the original vtk object,
`vtk.vtkPolyData <https://www.vtk.org/doc/nightly/html/classvtkPolyData.html>`_.



.. rubric:: Attributes

.. autoautosummary:: pyvista.PolyData
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.PolyData
   :methods:

.. autoclass:: pyvista.PolyData
   :show-inheritance:
   :members:
   :undoc-members:
