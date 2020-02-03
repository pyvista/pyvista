.. _jupyter_ref:

PyVista Jupyter Notebook Integration
------------------------------------

PyVista has an interface for visualizing plots in Jupyter.  The
``pyvista.PlotterITK`` class allows you interactively visualize a mesh
within a jupyter notebook.  For those who prefer plotting within
jupyter, this is an great way of visualizing using ``VTK`` and
``pyvista``.

Special thanks to thewtex
.. _itkwidgets: https://github.com/InsightSoftwareConsortium/itkwidgets


Installation
~~~~~~~~~~~~
To use `PlotterITK` you'll need to install ``itkwidgets>=0.25.2``.
Follow the installation steps here:
.. _itkwidgets: https://github.com/InsightSoftwareConsortium/itkwidgets#installation

You can install everything with `pip` if you prefer not using conda,
but be sure your juptyerlab is up-to-date.  If you encounter problems,
uninstall and reinstall jupyterlab using pip.


Example Plotting with ITKwidgets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following example shows how to create a simple plot that shows a
simple sphere.

.. code:: python

    import pyvista as pv

    # create a mesh and identify some scalars you wish to plot
    mesh = pv.Sphere()
    z = mesh.points[:, 2]

    # Plot using the ITKplotter
    pl = pv.PlotterITK()
    pl.add_mesh(mesh, scalars=z, smooth_shading=True)
    pl.show(True)


.. figure:: ../images/user-generated/itk_plotting_sphere.png
    :width: 600pt

    ITKwidgets with pyvista


For convenience, figures can also be plotted using the ``plot_itk`` function:

.. code:: python

    import pyvista as pv

    # create a mesh and identify some scalars you wish to plot
    mesh = pv.Sphere()
    z = mesh.points[:, 2]

    # Plot using the ITKplotter
    pv.plot_itk(mesh, scalars=z)


Additional binder examples can be found at:

.. _itkwidgets_binder: https://hub.gke.mybinder.org/user/insightsoftware-tium-itkwidgets-p2yw6xvh/lab

.. rubric:: Attributes

.. autoautosummary:: pyvista.PlotterITK
   :attributes:

.. rubric:: Methods

.. autoautosummary:: pyvista.PlotterITK
   :methods:

.. autoclass:: pyvista.PlotterITK
   :members:
   :undoc-members:
   :show-inheritance:
