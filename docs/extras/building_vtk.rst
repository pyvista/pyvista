.. _ref_building_vtk:

Building VTK
============
Kitware provides Python wheels for VTK at `PyPI VTK <https://pypi.org/project/vtk/>`_, but there are situations where you
may need to build VTK from source (e.g. new release of Python, EGL
rendering, additional features, etc).  As ``pyvista`` does not provide
``vtk``, you will have to build it manually.


Wheels on Linux and Mac OS
~~~~~~~~~~~~~~~~~~~~~~~~~~

Building VTK from source on Linux is fairly easy.  Using the default
build settings, build a Python wheel of VTK using ``ninja`` using the following script.  This script assumes Python 3.9, but you can use any modern Python version.
For some additional useful options, see the `conda-forge recipe <https://github.com/conda-forge/vtk-feedstock/blob/master/recipe/build.sh>`__.
Most of the ones below are designed to reduce the build time and resulting wheel size.

.. code-block:: bash

    git clone https://github.com/Kitware/VTK
    cd VTK

    mkdir build
    cd build
    PYBIN=/usr/bin/python3.9
    cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DVTK_BUILD_TESTING=OFF \
        -DVTK_BUILD_DOCUMENTATION=OFF -DVTK_BUILD_EXAMPLES=OFF -DVTK_DATA_EXCLUDE_FROM_ALL:BOOL=ON \
        -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=NO \
        -DVTK_WHEEL_BUILD=ON -DVTK_PYTHON_VERSION=3 -DVTK_WRAP_PYTHON=ON \
        -DPython3_EXECUTABLE=$PYBIN ../
    ninja
    $PYBIN setup.py bdist_wheel
    pip install dist/vtk-*.whl  # optionally install it

You may need to install ``python3.9-dev`` and ``ninja`` if you have
not already installed it.


.. _gpu_off_screen:

Off-Screen Plotting GPU Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VTK supports rendering with EGL, enabling rapid off-screen
using GPU hardware acceleration.  Unfortuantly, the default VTK wheels
are not built with this feature as it results in a > 400 MB wheel.
For the adventurous/desperate, build VTK with EGL for a given Python wheel on
Linux with the following:

You can build VTK for off-screen plotting using GPU support by modifying
the above ``cmake`` command with:

.. code-block:: bash

  cmake -GNinja \
    -DVTK_BUILD_TESTING=OFF \
    -DVTK_WHEEL_BUILD=ON \
    -DVTK_PYTHON_VERSION=3 \
    -DVTK_WRAP_PYTHON=ON \
    -DVTK_OPENGL_HAS_EGL=True \
    -DVTK_USE_X=False \
    -DPython3_EXECUTABLE=$PYBIN \
    ../
  ninja

  $PYBIN setup.py bdist_wheel
  $PYBIN -m pip install dist/vtk-*.whl  # optional

This disables any plotting using the X server, so be prepared to use
this module only on a headless display where you either intend to save
static images or stream the render window to another computer with a
display (e.g using ``use_ipyvtk=True`` and jupyterlab). In other words,
this wheel will make VTK unusable outside of an off-screen
environment, so only plan on installing it on a headless system
without a X server.


Building VTK on Windows
~~~~~~~~~~~~~~~~~~~~~~~
Please reference the directions at `Building VTK with Windows
<https://vtk.org/Wiki/VTK/Configure_and_Build#On_Windows_5>`_.  This
is generally a non-trivial process and is not for the faint-hearted.
