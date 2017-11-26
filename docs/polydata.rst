PolyData Surface
================
The ``vtkInterface.PolyData`` object adds addiction functionality to the vtk.vtkPolyData object, to include direct array access through numpy, one line plotting, and other mesh functions.


PolyData Creation
-----------------

Empty Object
~~~~~~~~~~~~
A polydata object can be initialized with:

.. code:: python

    import vtkInterface
    grid = vtkInterface.PolyData()

This creates an empty grid, and is not useful until points and cells are added to it.  VTK points and cells can be added with ``SetPoints`` and ``SetCells``, but the inputs to these need to be ``vtk.vtkCellArray`` and ``vtk.vtkPoints`` objects, which need to be populated with values.  Grid creation is simplified by initializing the grid directly from numpy arrays as in the following section.


Initialize from Numpy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A PolyData object can be created quickly from numpy arrays.  The vertex array contains the locations of the points of the mesh and the face array contains the number of points for each face and the indices of each of those faces.

.. code:: python
	  
    import numpy as np
    import vtkInterface

    # mesh points
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0]])

    # mesh faces
    faces = np.hstack([[4, 0, 1, 2, 3],  # square
                       [3, 0, 1, 4],     # triangle
                       [3, 1, 2, 4]])    # triangle

    surf = vtkInterface.PolyData(vertices, faces)

    # plot each face with a different color
    surf.Plot(scalars=np.arange(3))

.. image:: ./images/samplepolydata.png


Initialize from a File
~~~~~~~~~~~~~~~~~~~~~~
Both binary and ASCII .ply, .stl, and .vtk files can be read using vtkInterface.  For example, the vtkInterface package contains example meshes and these can be loaded with:

.. code:: python

    import vtkInterface
    from vtkInterface import examples
        
    # Load mesh
    mesh = vtkInterface.PolyData(examples.planefile)

This mesh can then be written to a vtk file using:

.. code:: python

    mesh.WriteMesh('plane.vtk')

These meshes are identical.

.. code:: python

    >>> import numpy as np
    >>> mesh_from_vtk = vtkInterface.LoadMesh('plane.vtk')    
    >>> print(np.allclose(mesh_from_vtk.points, mesh.points))
    True

    
Mesh Manipulation and Plotting
------------------------------
Meshes can be directly manipulated using numpy or with the built-in translation and rotation routines.  This example loads two meshes and moves, scales, and copies them.

.. code:: python

    import vtkInterface
    from vtkInterface import examples
    
    # load and shrink airplane
    airplane = vtkInterface.PolyData(examples.planefile)
    pts = airplane.GetNumpyPoints() # gets pointer to array
    pts /= 10 # shrink by 10x
    
    # rotate and translate ant so it is on the plane
    ant = vtkInterface.PolyData(examples.antfile)
    ant.RotateX(90)
    ant.Translate([90, 60, 15])
    
    # Make a copy and add another ant
    ant_copy = ant.Copy()
    ant_copy.Translate([30, 0, -10])

To plot more than one mesh a plotting class must be created to manage the plotting.  The following code creates the class and plots the meshes with various colors.

.. code:: python
    
    # Create plotting object
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(ant, 'r')
    plobj.AddMesh(ant_copy, 'b')

    # Add airplane mesh and make the color equal to the Y position.  Add a
    # scalar bar associated with this mesh
    plane_scalars = pts[:, 1]
    plobj.AddMesh(airplane, scalars=plane_scalars, stitle='Airplane Y\nLocation')

    # Add annotation text
    plobj.AddText('Ants and Plane Example')
    plobj.Plot()

.. image:: ./images/AntsAndPlane.png


vtkInterface.PolyData Grid Class Methods
----------------------------------------
The following is a description of the methods available to a ``vtkInterface.PolyData`` object.  It inherits all methods from the original vtk object, `vtk.vtkStructuredGrid <https://www.vtk.org/doc/nightly/html/classvtkPolyData.html>`_.

.. autoclass:: vtkInterface.PolyData
    :members:
