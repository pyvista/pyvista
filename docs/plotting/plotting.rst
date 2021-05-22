.. _plotting_ref:

Plotting
--------

When plotting with the interactive rendering windows in VTK, several keyboard
shortcuts are available:

+-------------------------------------+-----------------+-----------------------------------------------------+
| Key                                                   | Action                                              |
+=====================================+=================+=====================================================+
| Linux/Windows                       | Mac             |                                                     |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``q``                                                 | Close the rendering window                          |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``v``                                                 | Isometric camera view                               |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``w``                                                 | Switch all datasets to a `wireframe` representation |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``r``                                                 | Reset the camera to view all datasets               |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``s``                                                 | Switch all datasets to a `surface` representation   |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``shift+click`` or ``middle-click`` | ``shift+click`` | Pan the rendering scene                             |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``left-click``                      | ``cmd+click``   | Rotate the rendering scene in 3D                    |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``ctrl+click``                      |                 | Rotate the rendering scene in 2D (view-plane)       |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``mouse-wheel`` or ``right-click``  | ``ctl+click``   | Continuously zoom the rendering scene               |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``shift+s``                                           | Save a screenhsot (only on ``BackgroundPlotter``)   |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``shift+c``                                           | Enable interactive cell selection/picking           |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``up``/``down``                                       | Zoom in and out                                     |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``+``/``-``                                           | Increase/decrease the point size and line widths    |
+-------------------------------------+-----------------+-----------------------------------------------------+


Plotting in a Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Static and interactive inline plots are possible using a Jupyter
notebook.  The code snippet below will create a static screenshot of
the rendering and display it in the Jupyter notebook:


.. jupyter-execute::

    import pyvista as pv
    sphere = pv.Sphere()
    sphere.plot(jupyter_backend='static')


It is possible to use the ``Plotter`` class as well.

.. jupyter-execute::

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(sphere)
    plotter.show(jupyter_backend='static')

Additionally, you can generate interactive plots by leveraging a
jupyter plotting backend like ``panel`` or ``ipygany``.  You can even
use it to create interactive documentations online.

.. jupyter-execute::

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(sphere)
    plotter.show(jupyter_backend='ipygany')

For more details, see the section on :ref:`jupyter_plotting`.


Background Plotting
~~~~~~~~~~~~~~~~~~~

PyVista provides a plotter that enables users to create a rendering
window in the background that remains interactive while the user
performs their processing. This creates the ability to make a
rendering scene and interactively add or remove datasets from the
scene as well as has some useful menu functions for common scene
manipulation or export tasks. To get started, try instantiating the
:class:`pyvistaqt.BackgroundPlotter`:

.. code:: python

    import pyvista as pv
    import pyvistaqt as pvqt
    from pyvista import examples

    dataset = examples.load_hexbeam()

    p = pvqt.BackgroundPlotter()

    p.add_mesh(dataset)

    p.show_bounds(grid=True, location='back')



Plot Time Series Data
~~~~~~~~~~~~~~~~~~~~~

This example outlines how to plot data where the spatial reference and data
values change through time:


.. code-block:: python

    from threading import Thread
    import time
    import numpy as np
    import pyvista as pv
    import pyvistaqt as pvqt
    from pyvista import examples


    globe = examples.load_globe()
    globe.point_arrays['scalars'] = np.random.rand(globe.n_points)
    globe.set_active_scalars('scalars')


    plotter = pvqt.BackgroundPlotter()
    plotter.add_mesh(globe, lighting=False, show_edges=True, texture=True, scalars='scalars')
    plotter.view_isometric()

    # shrink globe in the background
    def shrink():
        for i in range(50):
            globe.points *= 0.95
            # Update scalars
            globe.point_arrays['scalars'] = np.random.rand(globe.n_points)
            time.sleep(0.5)

    thread = Thread(target=shrink)
    thread.start()


.. figure:: ../images/gifs/shrink-globe.gif


Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyvista.plot

.. autofunction:: pyvista.plot_arrows

.. autofunction:: pyvista.set_plot_theme

.. autofunction:: pyvista.create_axes_orientation_box


Base Plotter
~~~~~~~~~~~~

The base plotter class that all PyVista plotters inherit. Please note that the former
``BackgroundPlotter`` class has been moved to the `pyvistaqt`_ package.

.. _pyvistaqt: http://qtdocs.pyvista.org/


.. rubric:: Attributes

.. autoautosummary:: pyvista.BasePlotter
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.BasePlotter
   :methods:


.. autoclass:: pyvista.BasePlotter
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Plotter
~~~~~~~

.. rubric:: Attributes

.. autoautosummary:: pyvista.Plotter
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.Plotter
   :methods:

.. autoclass:: pyvista.Plotter
   :show-inheritance:
   :members:
   :undoc-members:


Renderer
~~~~~~~~

.. rubric:: Attributes

.. autoautosummary:: pyvista.Renderer
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.Renderer
   :methods:

.. autoclass:: pyvista.Renderer
   :show-inheritance:
   :members:
   :undoc-members:
