.. _plotting_ref:

Plotting
========

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




Convenience Functions
---------------------


.. autofunction:: vtki.plot


.. autofunction:: vtki.plot_arrows


.. autofunction:: vtki.set_plot_theme


Base Plotter
------------

The base plotter class that all ``vtki`` plotters inherit. Please note that the
:class:`vtki.BackgroundPlotter` is documented under :ref:`qt_ref`.


.. rubric:: Attributes

.. autoautosummary:: vtki.BasePlotter
   :attributes:

.. rubric:: Methods

.. autoautosummary:: vtki.BasePlotter
   :methods:


.. autoclass:: vtki.BasePlotter
   :show-inheritance:
   :members:
   :undoc-members:


Plotter
-------

.. rubric:: Attributes

.. autoautosummary:: vtki.Plotter
   :attributes:

.. rubric:: Methods

.. autoautosummary:: vtki.Plotter
   :methods:



.. autoclass:: vtki.Plotter
   :show-inheritance:
   :members:
   :undoc-members:


Renderer
--------

.. rubric:: Attributes

.. autoautosummary:: vtki.Renderer
   :attributes:

.. rubric:: Methods

.. autoautosummary:: vtki.Renderer
   :methods:



.. autoclass:: vtki.Renderer
   :show-inheritance:
   :members:
   :undoc-members:
