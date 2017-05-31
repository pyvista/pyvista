.. _overview_ref:

Overview
========

VTK is an excellent visualization toolkit, and with Python bindings it should 
be able to combine the speed of C++ with the rapid prototyping of Python.  
However, despite this VTK code programmed in Python generally looks the same 
as its C++ counterpart.  
For `example <http://www.vtk.org/Wiki/VTK/Examples/Python/STLReader>`_, loading
and plotting an STL file requires::


    import vtk
     
    filename = "myfile.stl"
     
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
     
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(reader.GetOutput())
    else:
        mapper.SetInputConnection(reader.GetOutputPort())
     
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
     
    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
     
    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
     
    # Assign actor to the renderer
    ren.AddActor(actor)
     
    # Enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()


The goal of vtkInterface is to simplfiy this without loosing functionality.  
The same stl can be loaded and plotted using vtkInterface with::

    import vtkInterface
    
    # Load mesh
    filename = "myfile.stl"
    mesh = vtkInterface.LoadMesh(filename)

    # Plot mesh interactively
    mesh.Plot()


When combined with numpy, you can make some truly spectacular plots::

    import vtkInterface
    import numpy as np
    
    # Make a grid
    x, y, z = np.meshgrid(np.linspace(-5, 5, 20),
                          np.linspace(-5, 5, 20),
                          np.linspace(-5, 5, 5))
    
    points = np.empty((x.size, 3))
    points[:, 0] = x.ravel('F')
    points[:, 1] = y.ravel('F')
    points[:, 2] = z.ravel('F')
    
    # Compute a direction for the vector field
    direction = np.sin(points)**3
    
    
    # plot using the plotting class
    plobj = vtkInterface.PlotClass()
    plobj.AddArrows(points, direction, 0.5)
    plobj.SetBackground([0, 0, 0]) # RGB set to black
    plobj.Plot()
    del plobj

.. image:: vectorfield.png


While not everything can be simplified without losing functionality, many of the
objects can.  For example, triangular surface meshes in VTK can be tesselated 
but every other object in VTK cannot be tesselated.  It then makes sense that
a tesselation method be added to the existing triangular surface mesh.  That
way, tesselation can be performed with::

    submesh = mesh.Subdivide('linear', nsub=3)

and help(mesh.Subdivide) yields a useful helpdoc::

    Help on function Subdivide in module vtkInterface.utilities:
    
    Subdivide(mesh, nsub, subfilter='linear')
        Increase the number of triangles in a single, connected triangular mesh.
        
        Uses one of the following vtk subdivision filters to subdivide a mesh.
        vtkButterflySubdivisionFilter
        vtkLoopSubdivisionFilter
        vtkLinearSubdivisionFilter
        
        Linear subdivision results in the fastest mesh subdivision, but it does not
        smooth mesh edges, but rather splits each triangle into 4 smaller 
        triangles.
        
        Butterfly and loop subdivision perform smoothing when dividing, and may
        introduce artifacts into the mesh when dividing.
        
        Subdivision filter appears to fail for multiple part meshes.  Should be one
        single mesh.
        
        Parameters
        ----------
        mesh : vtk.vtkPolyData
            Mesh to be subdivided.  Must be a triangular mesh.
        nsub : int
            Number of subdivisions.  Each subdivision creates 4 new triangles, so
            the number of resulting triangles is nface*4**nsub where nface is the
            current number of faces.
        subfilter : string, optional
            Can be one of the following: 'butterfly', 'loop', 'linear'
            
        Returns
        -------
        mesh : vtkPolydata object
            VTK surface mesh object.
            
        Examples
        --------
        >>> from vtkInterface import examples
        >>> import vtkInterface
            
        >>> mesh = vtkInterface.LoadMesh(examples.planefile)
        >>> submesh = mesh.Subdivide(1, 'loop')
    



