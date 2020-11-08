.. _qt_plotting:

Interactive Notebook Plotting Using ``ipyvtk-simple``
-----------------------------------------------------

``pyvista`` has the ability to display fully featured plots within a
jupyterlab enviornment using ``ipyvtk-simple``.  This feature works by
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
------------
If you're using an Anaconda enviornment, installation is the quite straightforward:

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

On Linux, you can setup your jupyterlab enviornment with:

.. code::

    pip install jupyterlab itkwidgets==0.32.0 ipywidgets=7.5.1 pyvista
    sudo apt install nodejs
    jupyter labextension install @jupyter-widgets/jupyterlab-manager@2.0 itkwidgets@0.32.0 ipycanvas@0.6.1 ipyevents@1.8.1



Performance Notes and Tweaks
----------------------------
Local usage on ``jupyterlab`` uses your GPU for off-screen rendering,
but on a headless instance (e.g. VM or kubernetes cluster), you're
limited to software rendering on the CPU.  This will result in
substancially slower renders and will appear quite slow (especially on
mybinder).

Fortuantly, VTK supports rendering with EGL, enabling rapid off-screen
using GPU hardware acceleration.  Unfortuantly, the default VTK wheels
are not built with this feature as it results in a > 400 MB wheel.
For the adventurous/desperate, build VTK with EGL for a given Python wheel on
Linux with the following:

.. code::

    PYBIN=/usr/bin/python3.8  # replace with your version...
    cmake -GNinja -DVTK_BUILD_TESTING=OFF -DVTK_WHEEL_BUILD=ON -DVTK_PYTHON_VERSION=3 -DVTK_WRAP_PYTHON=ON -DVTK_OPENGL_HAS_EGL=True -DVTK_USE_X=False -DPython3_EXECUTABLE=$PYBIN ../
    ninja

    $PYBIN setup.py bdist_wheel
    $PYBIN -m pip install dist/vtk-*.whl  # optional

Please note you'll need to have the ``-dev`` package installed for
Python along with ``ninja``.

Note that this wheel will make VTK unusable outside of an off-screen
enviornment, so only plan on installing it on a headless system
without a X server.

For those of you who 
