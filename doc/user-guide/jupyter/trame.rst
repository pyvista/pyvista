.. _trame_jupyter:

Trame Jupyter backend for PyVista
---------------------------------

PyVista has the ability to display fully featured plots within a
Jupyter environment using `Trame <https://kitware.github.io/trame/index.html>`.
We provide mechanisms to pair PyVista and Trame so that PyVista plotters
can be used in a web context with both server- and client-side rendering.

The server-side rendering mode of the trame backend works by streaming the
current render window to a canvas within Jupyter and then passing any user
actions from the canvas back to the VTK render window (this is done under
the hood by the ``vtkRemoteView`` in ``trame-vtk``.

For example, both sections of code will display an interactive canvas
within Jupyter:

.. code:: python

    import pyvista as pv
    sphere = pv.Sphere()

    # short example
    sphere.plot(jupyter_backend='trame')

    # long example
    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(sphere)
    plotter.show(jupyter_backend='trame')

For convenience, you can enable ``trame`` by default with:

.. note::
    It is critical to ``await`` the call to :func:`set_jupyter_backend() <pyvista.set_jupyter_backend>` when using trame in Jupyter.

.. code:: python

    import pyvista as pv
    await pv.set_jupyter_backend('trame')


Trame Jupyter Modes
+++++++++++++++++++

The PyVista Trame jupyter backend provides three modes of operation (technically
as three separate backend choices):

* ``'trame'``: Uses a view that can switch between client- and server-rendering modes.
* ``'server'``: Uses a view that is purely server-rendering.
* ``'client'``: Uses a view that is purely client-rendering (generally safe without a virtual frame buffer)

With any of these trame-based backend choices, you must await the call to
:func:`set_jupyter_backend() <pyvista.set_jupyter_backend>`
as mentioned above.

You can choose your backend either by using :func:`set_jupyter_backend() <pyvista.set_jupyter_backend>`
or passing ``jupyter_backend`` on the :func:`show() <pyvista.Plotter.show>` call.

.. code:: python

    import pyvista as pv
    await pv.set_jupyter_backend('client')

    pv.Cone().plot()


.. code:: python

    import pyvista as pv
    await pv.set_jupyter_backend('trame')

    pl = pv.Plotter()
    pl.add_mesh(pv.Cone())
    pl.show(jupyter_backend='client')


Installation
++++++++++++

Using pip, you can set up your jupyter environment with:

.. code::

    pip install 'jupyterlab>=3' ipywidgets 'pyvista[all,trame]'


Other Considerations
++++++++++++++++++++
It may be worth using GPU acceleration, see :ref:`gpu_off_screen`.

If you do not have GPU acceleration, be sure to start up a virtual
framebuffer using ``Xvfb``.  You can either start it using bash with:

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
``pyvista.start_xvfb()``.  Please be sure to install ``xvfb`` and
``libgl1-mesa-glx`` with:

.. code-block:: bash

    sudo apt-get install libgl1-mesa-dev xvfb
