.. _jupyter_plotting:

Jupyter Notebook Plotting
=========================
Plot with ``pyvista`` interactively within a `Juptyer
<https://jupyter.org/>`_ notebook!


Demo Using ``pythreejs``
~~~~~~~~~~~~~~~~~~~~~~~~
Create interactive physically based rendering using `pythreejs`_.

.. jupyter-execute::
   :hide-code:

   import pyvista
   pyvista.global_theme.background = 'white'
   pyvista.global_theme.antialiasing = True
   pyvista.global_theme.window_size = [600, 600]

.. jupyter-execute::

   import pyvista as pv
   from pyvista import examples

   # download an example and display it using physically based rendering.
   mesh = examples.download_lucy()
   mesh.plot(color='lightgrey', pbr=True, metallic=0.2,
             jupyter_backend='pythreejs')


Demo Using ``ipygany``
~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   from pyvista import demos

   # basic glyphs demo
   mesh = demos.glyphs(2)

   text = demos.logo.text_3d("I'm interactive!", depth=0.2)
   text.points *= 0.1
   text.translate([0, 1.4, 1.5])
   mesh += text
   mesh['Example Scalars'] = mesh.points[:, 0]

   mesh.plot(cpos='xy', jupyter_backend='ipygany', show_scalar_bar=True)


Demo Using ``panel``
~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   from pyvista import demos
   demos.plot_logo(jupyter_backend='panel')


Supported Modules
~~~~~~~~~~~~~~~~~
The PyVista module supports a variety of backends when plotting within
a jupyter notebook:

* Server-side rendering with PyVista streaming to the notebook through
  `ipyvtklink <https://github.com/Kitware/ipyvtklink/>`_
* Client-side rendering with `pythreejs`_ using ``threejs``.
* Client-side rendering with `ipygany <https://github.com/QuantStack/ipygany>`_ using ``threejs``.
* Client-side rendering using `panel <https://github.com/holoviz/panel>`_ using ``vtk.js``.
* Client-side rendering with `itkwidgets <https://github.com/InsightSoftwareConsortium/itkwidgets>`_ using ``itk.js`` and ``vtk.js``.
* Static images.

------------

Details for Each Backend
~~~~~~~~~~~~~~~~~~~~~~~~
See the individual package pages on each backend for additional
details on how to use these plotting backends.

.. toctree::
   :maxdepth: 1

   pythreejs
   ipygany
   panel
   ipyvtk_plotting
   itk_plotting


State of 3D Interactive Jupyterlab Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   3D plotting within Jupyter notebooks is an emerging technology,
   partially because Jupyter is still relatively new, but also because
   the web technology used here is also new and rapidly developing as
   more and more users and developers shift to the cloud or cloud-based
   visualization.  Things here are likely to break and rapidly change

   This was written in March 2021 and updated in August 2021, and may
   already be out of date.  Be sure to check the developer websites
   for any changes.

When plotting using Jupyterlab you have the option of using one of
many modules, each of which has its advantages, disadvantages, and
quirks.  While ``pyvista`` attempts to remove some of the differences
in the API when using the ``Plotting`` class, the plots will still
look and feel differently depending on the backend.  Additionally,
different backends have different requirements and may not support
your deployment environment.

This table details various capabilities and technologies used by the
jupyter notebook plotting modules:

+---------------+--------------+--------------------+---------+----------------------+
| Jupyter Notebook 3D Modules                                                        |
+---------------+--------------+--------------------+---------+----------------------+
|               | Jupyterlab 3 | Rendering Location | Backend | Requires Framebuffer |
+---------------+--------------+--------------------+---------+----------------------+
| panel         | Yes          | Client             | vtk.js  | Yes                  |
+---------------+--------------+--------------------+---------+----------------------+
| pythreejs     | Yes          | Client             | threejs | No                   |
+---------------+--------------+--------------------+---------+----------------------+
| ipygany       | Yes          | Client             | threejs | No                   |
+---------------+--------------+--------------------+---------+----------------------+
| ipyvtklink    | Yes          | Server             | vtk     | Yes                  |
+---------------+--------------+--------------------+---------+----------------------+
| itkwidgets    | No           | Client             | vtk.js  | Yes                  |
+---------------+--------------+--------------------+---------+----------------------+

At the moment, ``itkwidgets`` and ``ipyvtklink`` are incompatible with
Jupyterlab 3, and will result in a "Error displaying widget: model not
found" message from juptyer.  Additionally, all the modules other than
``ipygany`` and ``pythreejs`` require a framebuffer, which can be
setup on a headless environment with :func:`pyvista.start_xvfb`.
However, on Google Colab, where it's not possible to install system
packages, you should stick with a module like ``threejs``, which does
not require any server side rendering or framebuffer.

See :ref:`install_ref` for more details installing on a headless
environment for the backends requiring a framebuffer.  When installing
the individual packages, the Jupyterlab 3 compatible packages can be
installed with a simple ``pip install <package>``.  See the
installation instructions for the other packages for more details.


Usage with PyVista
~~~~~~~~~~~~~~~~~~
There are two ways to set the jupyter plotting backend.  First, it can
be done on a plot by plot basis by setting the ``jupyter_backend`` parameter in
either :func:`Plotter.show() <pyvista.Plotter.show>` or :func:`dataset.plot()
<pyvista.DataSet.plot>`.  You can also set it globally with the
:func:`pyvista.set_jupyter_backend`.  For further details:

.. autofunction:: pyvista.set_jupyter_backend


.. _pythreejs: https://github.com/jupyter-widgets/pythreejs
