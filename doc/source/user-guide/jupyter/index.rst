.. _jupyter_plotting:

Jupyter Notebook Plotting
=========================
Plot with ``pyvista`` interactively within a `Jupyter
<https://jupyter.org/>`_ notebook.

.. note::
   We recommend using the Trame-based backed. See :ref:`trame_jupyter`.


Demo Using ``pythreejs``
~~~~~~~~~~~~~~~~~~~~~~~~
Create interactive physically based rendering using `pythreejs`_.

.. jupyter-execute::
   :hide-code:

   import pyvista
   pyvista.global_theme.background = 'white'
   pyvista.global_theme.anti_aliasing = 'fxaa'
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
   text.translate([0, 1.4, 1.5], inplace=True)
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

* Server and client-side rendering with PyVista streaming to the notebook through
  `trame <https://github.com/Kitware/trame/>`_
* Client-side rendering with `pythreejs`_ using ``threejs``.
* Client-side rendering with `ipygany <https://github.com/QuantStack/ipygany>`_ using ``threejs``.
* Client-side rendering using `panel <https://github.com/holoviz/panel>`_ using ``vtk.js``.
* Static images.

------------

Details for Each Backend
~~~~~~~~~~~~~~~~~~~~~~~~
See the individual package pages on each backend for additional
details on how to use these plotting backends.

.. toctree::
   :maxdepth: 1

   trame
   pythreejs
   ipygany
   panel
   ipyvtk_plotting


State of 3D Interactive Jupyter Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   3D plotting within Jupyter notebooks is an emerging technology,
   partially because Jupyter is still relatively new, but also because
   the web technology used here is also new and rapidly developing as
   more and more users and developers shift to the cloud or cloud-based
   visualization. Things here are likely to break and rapidly change

   This was written in March 2021 and updated in January 2023, and may
   already be out of date. Be sure to check the developer websites
   for any changes.

When plotting using Jupyter you have the option of using one of
many modules, each of which has its advantages, disadvantages, and
quirks. While ``pyvista`` attempts to remove some of the differences
in the API when using the ``Plotting`` class, the plots will still
look and feel differently depending on the backend. Additionally,
different backends have different requirements and may not support
your deployment environment.

This table details various capabilities and technologies used by the
jupyter notebook plotting modules:

+---------------+--------------------+---------------+----------------------+
| Jupyter Notebook 3D Modules                                               |
+---------------+--------------------+---------------+----------------------+
|               | Rendering Location | Backend       | Requires Framebuffer |
+---------------+--------------------+---------------+----------------------+
| trame         | Client & Server    | vtk.js & vtk  | Optional             |
+---------------+--------------------+---------------+----------------------+
| panel         | Client             | vtk.js        | Yes                  |
+---------------+--------------------+---------------+----------------------+
| pythreejs     | Client             | threejs       | No                   |
+---------------+--------------------+---------------+----------------------+
| ipygany       | Client             | threejs       | No                   |
+---------------+--------------------+---------------+----------------------+

All the modules other than ``trame``, ``ipygany``, and ``pythreejs``
require a framebuffer, which can be set up on a headless environment
with :func:`pyvista.start_xvfb`.
However, on Google Colab, where it's not possible to install system
packages, you should stick with a module like ``threejs`` or the
``'client'`` variant of the trame-backend (see :ref:`trame_jupyter`),
which do not require any server side rendering or framebuffer.

See :ref:`install_ref` for more details installing on a headless
environment for the backends requiring a framebuffer. When installing
the individual packages, the Jupyterlab 3 compatible packages can be
installed with a simple ``pip install <package>``. See the
installation instructions for the other packages for more details.


Usage with PyVista
~~~~~~~~~~~~~~~~~~
There are two ways to set the jupyter plotting backend. First, it can
be done on a plot by plot basis by setting the ``jupyter_backend`` parameter in
either :func:`Plotter.show() <pyvista.Plotter.show>` or :func:`dataset.plot()
<pyvista.DataSet.plot>`. You can also set it globally with the
:func:`pyvista.set_jupyter_backend`. For further details:


.. code:: python

   import pyvista as pv
   pv.set_jupyter_backend('trame')

.. autofunction:: pyvista.set_jupyter_backend


.. _pythreejs: https://github.com/jupyter-widgets/pythreejs
