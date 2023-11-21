.. _jupyter_plotting:

Jupyter Notebook Plotting
=========================
Plot with ``pyvista`` interactively within a `Jupyter
<https://jupyter.org/>`_ notebook.

.. note::
   We recommend using the Trame-based backed. See :ref:`trame_jupyter`.


Supported Modules
~~~~~~~~~~~~~~~~~
The PyVista module supports a variety of backends when plotting within
a jupyter notebook:

* Server and client-side rendering with PyVista streaming to the notebook through
  `trame <https://github.com/Kitware/trame/>`_
* Static images.


.. toctree::
   :maxdepth: 1

   trame


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
