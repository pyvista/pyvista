Jupyter Notebook Plotting
=========================

.. jupyter-execute::

    import pyvista as pv
    from pyvista import examples

    mesh = examples.download_bunny()
    
    pl = pv.Plotter()
    pl.add_mesh(mesh, color='lightgrey')
    pl.background_color = 'white'
    pl.camera_position = 'xy'
    pl.show(jupyter_backend='ipygany')


The PyVista module supports a variety of backends when plotting within
a jupyter notebook:

* Server-side rendering with PyVista streaming to the notebook through ``ipyvtk_simple``
* Client-side rendering with ``itkwidgets``.
* Client-side rendering with ``ipygany``.
* Server and Client-side rendering with ``panel``.
* Static images.


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


