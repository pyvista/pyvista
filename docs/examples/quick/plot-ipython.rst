Plotting in a Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


.. figure:: ../../images/notebook_sphere.png
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



IPython Interactive Plotting Tools
----------------------------------

``vtki`` comes packed with several interactive plotting tools to make using the
filters a bit more intuitive (see :ref:`ipy_tools_ref`).
If in an IPython environment, call one of the tools on an input dataset to yield
widgets that will control a filter or task in an interactive rendering scene:

.. code:: python

   import vtki
   from vtki import examples

   dataset = examples.load_hexbeam()

   # Use the slicer tool
   vtki.OrthogonalSlicer(dataset)


.. figure:: https://github.com/akaszynski/vtki/raw/master/docs/images/slicer-tool.gif
  :width: 500pt
