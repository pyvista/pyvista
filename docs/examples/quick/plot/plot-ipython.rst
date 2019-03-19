Plotting in a Jupyter Notebook
------------------------------

Inline plots are possible using a Jupyter notebook.  The code snippet below
will create a static screenshot of the rendering and display it in the Jupyter
notebook:


.. code:: python

    import vtki
    sphere = vtki.Sphere()

    # short example
    cpos, image = sphere.plot(notebook=True)

    # long example
    plotter = vtki.Plotter(notebook=True)
    plotter.add_mesh(sphere)
    plotter.plot()


.. figure:: ../../../images/user-generated/notebook_sphere.png
    :width: 600pt

    Jupyter Inline Plotting

To display interactive plots in Jupyter notebooks, use the
:class:`vtki.BackgroundPlotter` to open a rendering window in the background
that you can manipulate in real time from the Jupyter notebook:

.. code-block:: python

    import vtki
    from vtki import examples

    dataset = examples.load_uniform()

    plotter = vtki.BackgroundPlotter()
    plotter.add_mesh(dataset)

    # Then in another cell, you can add more to the plotter
    plotter.add_bounds_axes()


Background Plotting
-------------------

``vtki`` provides a plotter that enables users to create a rendering window in
the background that remains interactive while the user performs their
processing. This creates the ability to make a rendering scene and interactively
add or remove datasets from the scene as well as has some useful menu functions
for common scene manipulation or export tasks. To get started, try instantiating
the :class:`vtki.BackgroundPlotter`:

.. code:: python

    import vtki
    from vtki import examples

    dataset = examples.load_hexbeam()

    p = vtki.BackgroundPlotter()

    p.add_mesh(dataset)

    p.add_bounds_axes(grid=True, location='back')


IPython Interactive Plotting Tools
----------------------------------

``vtki`` comes packed with several interactive plotting tools to make using the
filters a bit more intuitive (see :ref:`ipy_tools_ref`).
If in an IPython environment, call one of the tools on an input dataset to yield
widgets that will control a filter or task in an interactive rendering scene.
These tools create an :class:`vtki.BackgroundPlotter` instance which can be
accessed under the ``.plotter`` attribute for further scene manipulation:

.. code:: python

   import vtki
   from vtki import examples

   dataset = examples.load_hexbeam()

   # Use the slicer tool
   tool = vtki.OrthogonalSlicer(dataset)

   # Get the plotter for adding more datasets:
   p = tool.plotter
   p.show_grid()


.. figure:: ../../../images/gifs/slicer-tool.gif
  :width: 500pt
