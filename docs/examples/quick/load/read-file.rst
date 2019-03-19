Load and Plot from a File
-------------------------

Loading a mesh is trivial.  The following code block uses a built-in example
file, displays an airplane mesh, saves a screenshot, and returns the camera's
position:

.. testcode:: python

    import vtki
    from vtki import examples
    import numpy as np

    filename = examples.planefile
    mesh = vtki.read(filename)
    cpos = mesh.plot(screenshot='./images/airplane.png', color='orange')

.. image:: ../../../images/airplane.png


You can also take a screenshot without creating an interactive plot window using
the ``Plotter``:

.. testcode:: python

    plotter = vtki.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color='orange')
    plotter.plot(auto_close=False)
    plotter.screenshot('./images/airplane.png')
    plotter.close()

The ``img`` array can be used to plot the screenshot in ``matplotlib``:

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

If you need to setup the camera you can do this by plotting first and getting
the camera after running the ``plot`` function:

.. testcode:: python

    plotter = vtki.Plotter()
    plotter.add_mesh(mesh)
    cpos = plotter.plot()

You can then use this cached camera for additional plotting without having to
manually interact with the ``vtk`` plot window:

.. testcode:: python

    plotter = vtki.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color='orange')
    plotter.camera_position = cpos
    plotter.plot(auto_close=False)
    plotter.screenshot('./images/airplane.png')
    plotter.close()

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

    >>> print(mesh.faces.reshape(-1, 4)[:, 1:])

     [[   0    1    2]
      [   0    2    3]
      [   4    5    1]
      ...,
      [1324 1333 1323]
      [1325 1216 1334]
      [1325 1334 1324]]
