.. _ipyvtk_plotting:

Using ``ipyvtklink`` with PyVista
---------------------------------

.. deprecated:: 0.38.0
   This backend has been deprecated in favor of :ref:`trame_jupyter` - a new
   framework for building dynamic web applications with Python with great
   support for VTK.

``pyvista`` has the ability to display fully featured plots within a
JupyterLab environment using ``ipyvtklink``.  This feature works by
streaming the current render window to a canvas within JupyterLab and
then passing any user actions from the canvas back to the VTK render
window.

While this isn't an exciting feature when JupyterLab is being run
locally, this has huge implications when plotting remotely as you can
display any plot, with subplots and widgets, from JupyterLab.

For example, both sections of code will display an interactive canvas
within JupyterLab:

.. code:: python

    import pyvista as pv
    sphere = pv.Sphere()

    # short example
    image = sphere.plot(jupyter_backend='ipyvtklink', return_cpos=False)

    # long example
    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(sphere)
    plotter.show(jupyter_backend='ipyvtklink')

For convenience, you can enable ``ipyvtklink`` by default with:

.. code:: python

    import pyvista
    pyvista.set_jupyter_backend('ipyvtklink')


Installation
++++++++++++
If you're using an Anaconda environment, installation is the quite straightforward:

.. code::

    conda env create --name pyvista --file environment.yml

Where environment.yml is:

.. code::

    channels:
      - conda-forge
      - defaults
    dependencies:
      - jupyterlab >=3
      - ipywidgets
      - pyvista
      - ipyvtklink
      - matplotlib

Using pip, you can set up your jupyterlab environment with:

.. code::

    pip install 'jupyterlab>=3' ipywidgets 'pyvista[all]' ipyvtklink



Other Considerations
++++++++++++++++++++
It may be worth using GPU acceleration, see :ref:`gpu_off_screen`.

If you do not have GPU acceleration, be sure to start up a virtual
framebuffer using ``Xvfb``.  You can either start it using bash with:

.. code-block:: bash

    export DISPLAY=:99.0
    export PYVISTA_OFF_SCREEN=true
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
