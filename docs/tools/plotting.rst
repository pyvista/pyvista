.. _plotting_ref:

Plotting
========

When plotting with the interactive rendering windows in VTK, several keyboard
shortcuts are available:

+-----------------+-----------------------------------------------------+
| Key             | Action                                              |
+-----------------+-----------------------------------------------------+
| ``q``           | Close the rendering window                          |
+-----------------+-----------------------------------------------------+
| ``w``           | Switch all datasets to a `wireframe` representation |
+-----------------+-----------------------------------------------------+
| ``r``           | Reset the camera to view all datasets               |
+-----------------+-----------------------------------------------------+
| ``s``           | Switch all datasets to a `surface` representation   |
+-----------------+-----------------------------------------------------+
| ``shift+click`` | Pan the rendering scene                             |
+-----------------+-----------------------------------------------------+
| ``cmd+click``   | Rotate the rendering scene                          |
+-----------------+-----------------------------------------------------+
| ``ctl+click``   | Continuously zoom the rendering scene               |
+-----------------+-----------------------------------------------------+




Convenience Functions
---------------------


.. autofunction:: vtki.plot


.. autofunction:: vtki.plot_arrows


.. autofunction:: vtki.set_plot_theme


Base Plotter
------------

The base plotter class that all ``vtki`` plotters inherit. Please note that the
:class:`vtki.BackgroundPlotter` is documented under :ref:`qt_ref`.


.. autoclass:: vtki.BasePlotter
   :show-inheritance:
   :members:
   :undoc-members:


Plotter
-------

.. autoclass:: vtki.Plotter
   :show-inheritance:
   :members:
   :undoc-members:
