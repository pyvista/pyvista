Jupyter Notebook Plotting
=========================

Plot with ``pyvista`` interactively within a `juptyer
<https://jupyter.org/>`_ notebook!

.. jupyter-execute::

    from pyvista import demos

    mesh = demos.glyphs(2)

    text = demos.logo.text_3d("I'm interactive!", depth=0.1)
    text.points *= 0.1
    text.translate([0, 1.4, 1.5])
    mesh += text
    mesh['x'] = mesh.points[:, 0] * mesh.points[:, 2]

    mesh.plot(cpos='xy', jupyter_backend='ipygany', background='white',
              show_scalar_bar=False)


Supported Modules
~~~~~~~~~~~~~~~~~
The PyVista module supports a variety of backends when plotting within
a jupyter notebook:

* Server-side rendering with PyVista streaming to the notebook through
  ``ipyvtk_simple``.
* Client-side rendering with ``ipygany`` using ``threejs``.
* Client-side rendering using ``panel`` using ``vtk.js``.
* Client-side rendering with ``itkwidgets`` using ``itklibraries``.
* Static images.

State of 3D Interactive Jupyterlab Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   This

Usage
~~~~~
There are two ways setting the jupyter plotting backend.  First, it
can be done on a plot by plot basis by setting the ``jupyter_backend``
in either ``mesh.plot()`` or ``plotter.show()``.  You can also set it
globally with the ``pyvista.set_jupyter_backend`` function:

.. autofunction:: pyvista.set_jupyter_backend

.. toctree::
   :maxdepth: 2

   ipyvtk_plotting
   itk_plotting
   ipygany


