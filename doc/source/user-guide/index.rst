User Guide
**********
This section details the general usage of PyVista for users who may or
may not have used VTK in the past, but are looking to leverage it in a
Pythonic manner for 3D plotting. See the table of contents below or
the in the side panel for the individual sections demonstrating the
key concepts of PyVista.


Simple Interactive Example
~~~~~~~~~~~~~~~~~~~~~~~~~~
This basic example demonstrates three key features of PyVista:

- Simple `numpy`_ and `matplotlib`_ like interface
- Variety of built-in examples
- Intuitive plotting with keyword arguments.

.. _numpy: https://numpy.org/
.. _matplotlib: https://matplotlib.org/

.. pyvista-plot::
   :include-source: false
   :nofigs:
   :context:

   import pyvista as pv
   pv.set_jupyter_backend('static')
   pv.global_theme.background = 'white'
   pv.global_theme.window_size = [600, 400]
   pv.global_theme.axes.show = False
   pv.global_theme.smooth_shading = True
   pv.global_theme.anti_aliasing = 'fxaa'


Here, we download the `Stanford dragon mesh
<http://graphics.stanford.edu/data/3Dscanrep/>`_, color it according
to height, and plot it using a web-viewer. This same example will run
identically locally.

.. pyvista-plot::
    :context:

    >>> from pyvista import examples
    >>> mesh = examples.download_dragon()
    >>> mesh['scalars'] = mesh.points[:, 1]
    >>> mesh.plot(cpos='xy', cmap='plasma')


With just a few lines of code we downloaded a sample mesh from the
web, added scalars to it based on the points of the mesh, and plotted
it while controlling the orientation, color, and data presented in the
visualization.

The following sections explain the details of the how and why of
PyVista's interface.

User Guide Contents
===================

.. toctree::
   :maxdepth: 2

   what-is-a-mesh
   simple
   data_model
   vtk_to_pyvista
   themes
   jupyter/index
   optional_features

Videos
======
Here are some videos that you can watch to learn PyVista:

- PyConJP2020 talk "How to plot unstructured mesh file on Jupyter
  Notebook" (15 minutes):

  - `Video <https://youtu.be/X3Z54Kw4I6Y>`_
  - `Material <https://docs.google.com/presentation/d/1M_cnS66ja81u_mHACjaUsDj1wSeeEtnEevk_IMZ8-dg/edit?usp=sharing>`_

- Software Underground Tutorial

  - `Tutorial: PyVista <https://www.youtube.com/watch?v=FmNmRBsEBHE>`_

If there is any material that we can add, please open an `issue
<https://github.com/pyvista/pyvista/issues>`_ .
