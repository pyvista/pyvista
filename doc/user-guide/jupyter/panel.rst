.. _panel_ref:

Using ``Panel`` with PyVista
----------------------------
PyVista supports the usage of the `panel
<https://github.com/holoviz/panel>`_ module as a ``vtk.js`` jupyterlab
plotting backend that can be utialized as either a standalone VTK
viewer, or as a tightly integrated ``pyvista`` plotting backend.  For
example, within a Jupyter notebook environment, you can pass
``jupyter_backend='panel'`` to ``plot``, or ``Plotter.show`` to
automatically enable plotting with Juptyer and ``panel``.

For example, here's the ``PyVista`` logo:

.. jupyter-execute::

   from pyvista import demos
   demos.plot_logo(background='white', jupyter_backend='panel')

Note that this isn't a perfect replica since there are some details
lost in the conversion to ``vtk.js``, but for the vast majority of
cases, this can be used to accurately render ``pyvista`` plots within
Jupyterlab.


Examples and Usage
~~~~~~~~~~~~~~~~~~
There are two ways to use ``panel`` within Jupyter notebooks.  It can
be done on a plot by plot basis by setting the ``jupyter_backend`` in
``mesh.plot()``:

.. jupyter-execute::

    import pyvista as pv
    from pyvista import examples

    # create a point cloud from lidar data and add height scalars
    dataset = examples.download_lidar()
    point_cloud = pv.PolyData(dataset.points[::100])
    point_cloud['height'] = point_cloud.points[:, 2]
    point_cloud.plot(window_size=[500, 500],
                     jupyter_backend='panel',
                     cmap='jet',
                     point_size=2,
                     background='w')


Alternatively, you can set the backend globally:

.. jupyter-execute::

    import math
    import numpy
    import numpy as np

    import pyvista
    from pyvista import examples

    # set the global jupyterlab backend.  All plots from this point
    # onward will use the ``panel`` backend and do not have to be
    # specified in ``show``
    pyvista.set_jupyter_backend('panel')

    # create a sphere for Mars
    sphere = pyvista.Sphere(radius=1, theta_resolution=90, phi_resolution=90,
                            start_theta=270.001, end_theta=270)
    sphere.active_t_coords = numpy.zeros((sphere.points.shape[0], 2))

    sphere.active_t_coords[:, 0] = 0.5 + np.arctan2(-sphere.points[:, 0], sphere.points[:, 1])/(2 * math.pi)
    sphere.active_t_coords[:, 1] = 0.5 + np.arcsin(sphere.points[:, 2]) / math.pi

    tex = pyvista.read_texture(examples.download_mars_jpg())

    # with a black background
    pl = pyvista.Plotter(window_size=[500, 500])
    pl.background_color = 'black'
    pl.add_mesh(sphere, texture=tex, smooth_shading=False)
    pl.show()


Configuration Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If running on a headless environment (e.g. Google Colab, your own VM),
be sure to start up a virtual framebuffer using ``Xvfb``.  You can
either start it using bash with:

.. code-block:: bash

    export DISPLAY=:99.0
    export PYVISTA_OFF_SCREEN=true
    export PYVISTA_USE_IPYVTK=true
    which Xvfb
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 3
    set +x
    exec "$@"


Or alternatively, start it using the built in
``pyvista.start_xvfb()``.  Be sure to install ``xvfb`` and
``libgl1-mesa-glx`` with:

.. code-block:: bash

    sudo apt-get install libgl1-mesa-dev xvfb

Or using the package manager used by your environment.
