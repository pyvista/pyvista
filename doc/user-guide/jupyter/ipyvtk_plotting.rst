.. _ipyvtk_plotting:

Using ``ipyvtklink`` with PyVista
---------------------------------

.. note::
   As of version ``0.1.4``, ``ipyvtklink`` does not support
   Jupyterlab 3.  Attempting to run the following will return a
   ``Model not found`` error within jupyterlab.

``pyvista`` has the ability to display fully featured plots within a
JupyterLab environment using ``ipyvtklink``.  This feature works by
streaming the current render window to a canvas within JupyterLab and
then passing any user actions from the canvas back to the VTK render
window.

While this isn't an exciting feature when JupyterLab is being run
locally, this has huge implications when plotting remotely as you can
display any plot (except for those with multiple render windows) from
JupyterLab.

For example, both sections of code will display an interactive canvas
within JupyterLab:

.. code:: python

    import pyvista as pv
    sphere = pv.Sphere()

    # short example
    image = sphere.plot(use_ipyvtk=True, return_cpos=False)

    # long example
    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(sphere)
    plotter.show(jupyter_backend='ipyvtklink')

For convenience, you can enable ``ipyvtklink`` by default with:

.. code:: python

    import pyvista
    pyvista.global_theme.jupyter_backend = 'ipyvtklink'


Installation
++++++++++++
If you're using an Anaconda environment, installation is the quite straightforward:

.. code::

    conda env update --name base --file environment.yml
    conda run conda install -y nodejs
    conda run jupyter labextension install @jupyter-widgets/jupyterlab-manager@2.0 itkwidgets@0.32.0 ipycanvas@0.6.1 ipyevents@1.8.1

Where environment.yml is:

.. code::

    channels:
      - conda-forge
      - defaults
    dependencies:
      - jupyterlab=2.2.9
      - itkwidgets=0.32.0
      - ipywidgets=7.5.1
      - pyvista=0.27.0

On Linux, you can setup your jupyterlab environment with:

.. code::

    pip install jupyterlab itkwidgets==0.32.0 ipywidgets=7.5.1 pyvista
    sudo apt install nodejs
    jupyter labextension install @jupyter-widgets/jupyterlab-manager@2.0 itkwidgets@0.32.0 ipycanvas@0.6.1 ipyevents@1.8.1



Other Considerations
~~~~~~~~~~~~~~~~~~~~
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
