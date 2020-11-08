Building VTK
============
Kitware provides Python wheels for VTK at `PyPi VTK <https://pypi.org/project/vtk/>`_, but there are situations where you
may need to build VTK from source (e.g. new release of Python, EGL
rendering, additional features, etc).  As ``pyvista`` does not provide
``vtk``, you will have to build it manually.


Building VTK Python Wheels on Linux and Mac OS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building VTK from source on Linux is fairly easy.  Using the default
build settings, build a Python wheel of VTK using ``ninja`` using the following script.  This script assumes Python 3.9, but you can use any modern Python version.

.. code::

    git clone https://github.com/Kitware/VTK
    cd VTK

    mkdir build
    cd build
    PYBIN=/usr/bin/python3.9
    cmake -GNinja -DVTK_BUILD_TESTING=OFF -DVTK_WHEEL_BUILD=ON -DVTK_PYTHON_VERSION=3 -DVTK_WRAP_PYTHON=ON -DPython3_EXECUTABLE=$PYBIN ../

    ninja
    $PYBIN setup.py bdist_wheel
    pip install dist/vtk-*.whl  # optionally install it

You may need to install ``python3.9-dev`` and ``ninja`` if you have
not already installed it.


Building VTK on Windows
~~~~~~~~~~~~~~~~~~~~~~~
Please reference the directions at `Building VTK with Windows
<https://vtk.org/Wiki/VTK/Configure_and_Build#On_Windows_5>`_.  This
is generally a non-trivial process and is not for the faint-hearted.
