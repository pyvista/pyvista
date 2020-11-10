.. _ipyvtk_plotting:

Using ``ipyvtk-simple``
~~~~~~~~~~~~~~~~~~~~~~~

``pyvista`` has the ability to display fully featured plots within a
jupyterlab environment using ``ipyvtk-simple``.  This feature works by
streaming the current render window to a canvas within jupyterlab and
then passing any user actions from the canvas back to the VTK render
window.

While this isn't an exciting feature when jupyterlab is being run
locally, this has huge implications when plotting remotely as you can
display any plot (except for those with multiple render windows) from
jupyterlab.

For example, both sections of code will display an interactive canvas
within jupyterlab:

.. code:: python

    import pyvista as pv
    sphere = pv.Sphere()

    # short example
    cpos, image = sphere.plot(use_ipyvtk=True)

    # long example
    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(sphere)
    plotter.show(use_ipyvtk=True)

For convenience, you can enable ``use_ipyvtk`` by default with:

.. code:: python

    import pyvista
    pyvista.rcParams['use_ipyvtk'] = True


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
--------------------

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


Or alternatively, start it using the built in ``pyvista.start_xvfb()``.  Please be sure to install ``xvfb`` and ``libgl1-mesa-glx`` with:

.. code-block:: bash

    sudo apt-get install libgl1-mesa-dev xvfb
