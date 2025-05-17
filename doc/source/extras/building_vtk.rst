.. _building_vtk:

Building VTK
============
Kitware provides Python wheels for VTK at `PyPI VTK
<https://pypi.org/project/vtk/>`_, but there are situations where you
may need to build VTK from source (for example, a new release of Python, EGL
rendering, additional features, etc). As ``pyvista`` does not provide
``vtk``, you will have to either build it manually or install the default
wheel from PyPI.

.. note::
   Should you need a prebuilt wheel, a variety of prebuilt wheels can be found at
   `pyvista-wheels <https://github.com/pyvista/pyvista-wheels>`_, but you may be
   better off building your own. These are not "official" wheels and will soon
   be removed in favor of more "official" wheel variants from VTK directly.

Reference the official directions for `Building VTK
<https://gitlab.kitware.com/vtk/vtk/-/blob/master/Documentation/dev/build.md>`_.
The following directions assume you want to build a Python wheel non-standard
situations like EGL.


Building Wheels
~~~~~~~~~~~~~~~
Building VTK from source is fairly straightforward. Using the default build
settings, build a Python wheel of VTK using ``ninja`` using the following
script. This script uses system python3, but you can use any modern Python
version. For some additional useful options, see the `conda-forge recipe
<https://github.com/conda-forge/vtk-feedstock/blob/main/recipe/build-base.sh>`__.
Most of the ones below are designed to reduce the build time and resulting
wheel size.

.. note::
   We have also published some convenient CMake configurations files that you
   can adopt from `banesullivan/vtk-cmake <https://github.com/banesullivan/vtk-cmake>`_. These configurations cover the build variants described here
   and make the process of reproducibly building VTK wheel variants more
   straightforward.

.. code-block:: bash

    #!/bin/bash

    # Install build dependencies

    # Linux/Debian
    sudo add-apt-repository -y ppa:deadsnakes/ppa  # if on Ubuntu 20.04 or 18.04, building 3.10 for example
    sudo apt update
    sudo apt install -y ninja-build cmake libgl1-mesa-dev python3-dev git
    sudo apt install -y python3.10-dev python3.10-distutils  # if using deadsnakes + Python 3.10
    # If on 18.04, you'll need a newer cmake. You can follow VTK's instructions @ https://apt.kitware.com

    # Linux/CentOS
    sudo yum install epel-release
    sudo yum install ninja-build cmake mesa-libGL-devel mesa-libGLU-devel

    git clone https://gitlab.kitware.com/vtk/vtk.git
    mkdir vtk/build
    cd vtk/build
    git checkout v9.1.0  # optional to select a version, but recommended

    export PYBIN=/usr/bin/python3.10  # select your version of choice
    cmake -GNinja \
          -DCMAKE_BUILD_TYPE=Release \
          -DVTK_BUILD_TESTING=OFF \
          -DVTK_BUILD_DOCUMENTATION=OFF \
          -DVTK_BUILD_EXAMPLES=OFF \
          -DVTK_DATA_EXCLUDE_FROM_ALL:BOOL=ON \
          -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=NO \
          -DVTK_MODULE_ENABLE_VTK_WebCore:STRING=YES \
          -DVTK_MODULE_ENABLE_VTK_WebGLExporter:STRING=YES \
          -DVTK_MODULE_ENABLE_VTK_WebPython:STRING=YES \
          -DVTK_WHEEL_BUILD=ON \
          -DVTK_PYTHON_VERSION=3 \
          -DVTK_WRAP_PYTHON=ON \
          -DVTK_OPENGL_HAS_EGL=False \
          -DPython3_EXECUTABLE=$PYBIN ../
    ninja

    # build wheel in dist
    $PYBIN -m pip install wheel
    $PYBIN setup.py bdist_wheel
    $PYBIN -m pip install dist/vtk-*.whl  # optionally install it

.. _gpu_off_screen:


Off-Screen Plotting GPU Support
+++++++++++++++++++++++++++++++
VTK supports rendering with EGL, enabling rapid off-screen rendering
using GPU hardware acceleration without installing a virtual
framebuffer. The default VTK wheels are not built with this feature,
but you can build VTK for off-screen plotting using GPU support by
modifying the above ``cmake`` command with:

.. code-block:: bash

   #!/bin/bash

   # install build dependencies (Linux/Debian)
   apt-get update
   apt-get install -y ninja-build cmake libegl1-mesa-dev python3-dev

   # build using EGL
   git clone https://github.com/Kitware/VTK
   mkdir VTK/build
   cd VTK/build \
   git checkout v9.1.0
   cd /VTK/build
   cmake -GNinja \
     -DCMAKE_BUILD_TYPE=Release \
     -DVTK_BUILD_TESTING=OFF \
     -DVTK_BUILD_DOCUMENTATION=OFF \
     -DVTK_BUILD_EXAMPLES=OFF \
     -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=NO \
     -DVTK_MODULE_ENABLE_VTK_WebCore:STRING=YES \
     -DVTK_MODULE_ENABLE_VTK_WebGLExporter:STRING=YES \
     -DVTK_MODULE_ENABLE_VTK_WebPython:STRING=YES \
     -DVTK_WHEEL_BUILD=ON \
     -DVTK_PYTHON_VERSION=3 \
     -DVTK_WRAP_PYTHON=ON \
     -DVTK_OPENGL_HAS_EGL:BOOL=ON \
     -DVTK_USE_X:BOOL=OFF \
     -DVTK_USE_COCOA:BOOL=OFF \
     -DVTK_DEFAULT_RENDER_WINDOW_HEADLESS:BOOL=ON \
     -DPython3_EXECUTABLE=/usr/bin/python3 ../
   ninja

   # build the python wheel
   python3 -m pip install wheel \
   python3 setup.py bdist_wheel \
   pip install dist/vtk-*.whl

This disables any plotting using the X server, so be prepared to use
this module only on a headless display where you either intend to save
static images or stream the render window to another computer with a
display (e.g using ``pyvista.set_jupyter_backend('server')`` and
jupyterlab). In other words, this wheel will make VTK unusable outside
of an off-screen environment, so only plan on installing it on a
headless system without an X server.


Building OSMesa
+++++++++++++++
OSMesa provides higher visualization performance on CPU based hosts. Use this
instead of ``xvfb``:

.. code-block:: bash

   sudo apt-get install libosmesa6-dev cmake ninja-build

   git clone https://github.com/Kitware/VTK.git
   cd VTK
   git checkout v9.1.0
   mkdir build
   cd build

   PYBIN=/usr/bin/python
   cmake -GNinja \
         -DCMAKE_BUILD_TYPE=Release \
         -DVTK_BUILD_TESTING=OFF \
         -DVTK_BUILD_DOCUMENTATION=OFF \
         -DVTK_BUILD_EXAMPLES=OFF \
         -DVTK_DATA_EXCLUDE_FROM_ALL:BOOL=ON \
         -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=NO \
         -DVTK_MODULE_ENABLE_VTK_WebCore:STRING=YES \
         -DVTK_MODULE_ENABLE_VTK_WebGLExporter:STRING=YES \
         -DVTK_MODULE_ENABLE_VTK_WebPython:STRING=YES \
         -DVTK_WHEEL_BUILD=ON \
         -DVTK_PYTHON_VERSION=3 \
         -DVTK_WRAP_PYTHON=ON \
         -DVTK_OPENGL_HAS_EGL=False \
         -DVTK_OPENGL_HAS_OSMESA=True \
         -DVTK_USE_COCOA=FALSE \
         -DVTK_USE_X=FALSE \
         -DVTK_DEFAULT_RENDER_WINDOW_HEADLESS=True \
         -DPython3_EXECUTABLE=$PYBIN ../
   ninja
   $PYBIN setup.py bdist_wheel

Wheels will be generated in the ``dist`` directory.


Building ManyLinux Wheels
+++++++++++++++++++++++++
The above directions are great for building a local build of VTK, but
these wheels are difficult to share outside your local install given
issues with ABI compatibility due to the version of Linux they were
built on. You can work around this by building your wheels using a
`manylinux <https://github.com/pypa/manylinux>`_ docker image.

To do this, create a ``build_wheels.sh`` with the following contents in the
``git clone`` d ``vtk`` directory, and give it executable permissions
(``chmod +x build_wheels.sh``):

.. code-block:: bash

    #!/bin/bash
    # builds python wheels on docker container and tests installation

    set -e -x

    # build based on python version from args
    PYTHON_VERSION="$1"
    case $PYTHON_VERSION in
    3.9)
      PYBIN="/opt/python/cp39-cp39/bin/python"
      ;;
    3.10)
      PYBIN="/opt/python/cp310-cp310/bin/python"
      ;;
    3.11)
      PYBIN="/opt/python/cp311-cp311/bin/python"
      ;;
    esac

    yum install -y ninja-build cmake mesa-libGL-devel mesa-libGLU-devel

    rm -rf /io/build
    mkdir /io/build -p
    cd /io/build

    cmake -GNinja \
          -DCMAKE_BUILD_TYPE=Release \
          -DVTK_BUILD_TESTING=OFF \
          -DVTK_BUILD_DOCUMENTATION=OFF \
          -DVTK_BUILD_EXAMPLES=OFF \
          -DVTK_DATA_EXCLUDE_FROM_ALL:BOOL=ON \
          -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=NO \
          -DVTK_MODULE_ENABLE_VTK_WebCore:STRING=YES \
          -DVTK_MODULE_ENABLE_VTK_WebGLExporter:STRING=YES \
          -DVTK_MODULE_ENABLE_VTK_WebPython:STRING=YES \
          -DVTK_WHEEL_BUILD=ON \
          -DVTK_PYTHON_VERSION=3 \
          -DVTK_WRAP_PYTHON=ON \
          -DVTK_OPENGL_HAS_EGL=False \
          -DPython3_EXECUTABLE=$PYBIN ../
    ninja-build

    # build wheel in dist
    rm -rf dist
    $PYBIN -m pip install wheel
    $PYBIN setup.py bdist_wheel

    # cleanup wheel
    rm -rf wheelhouse
    auditwheel repair dist/*.whl

This script can then be called with:

.. code-block:: bash

    export PYTHON_VERSION=3.10
    docker run --cpus 4.5 -e \
           --rm -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 \
           /io/build_wheels.sh $PYTHON_VERSION

You should end up with a ``build/wheelhouse/vtk-*.whl``.

.. note::
   To build the EGL version of the wheel, follow the directions in the
   previous section. Add ``mesa-libEGL-devel`` to the installation
   dependencies.


Building Python VTK Wheel on Raspberry Pi (64-bit)
++++++++++++++++++++++++++++++++++++++++++++++++++
While it's possible to build on 32-bit Raspberry Pi (ARMv7), there are
several issues that crop up when building wheels for the 32-bit
version (see `manylinux issue 84
<https://github.com/pypa/manylinux/issues/84>`_). Should you attempt
to build on 32-bit, try building the wheel using `dockcross
<https://github.com/dockcross/dockcross>`_ as you may run into memory
limitations otherwise (especially with only 1 GB RAM).

Building the ``aarch64`` manylinux wheel can be done via docker with
the ``quay.io/pypa/manylinux2014_aarch64`` image. Run the following:

.. code-block:: bash

    PYTHON_VERSION=3.9
    rm -rf build
    docker run -e \
           --rm -v `pwd`:/io quay.io/pypa/manylinux2014_aarch64 \
           /io/build_wheels.sh $PYTHON_VERSION

Where ``build_wheels.sh`` is:

.. code-block:: bash

    #!/bin/bash
    # builds python wheels on docker container and tests installation

    set -e -x

    # build based on python version from args
    PYTHON_VERSION="$1"
    case $PYTHON_VERSION in
    3.9)
      PYBIN="/opt/python/cp39-cp39/bin/python"
      ;;
    3.10)
      PYBIN="/opt/python/cp310-cp310/bin/python"
      ;;
    3.11)
      PYBIN="/opt/python/cp311-cp311/bin/python"
      ;;
    esac

    /bin/bash
    yum install epel-release
    yum install ninja-build
    yum install mesa-libEGL-devel  # only needed when building EGL

    mkdir /io/build -p
    cd /io/build

    cmake -GNinja \
          -DCMAKE_BUILD_TYPE=Release \
          -DVTK_BUILD_TESTING=OFF \
          -DVTK_BUILD_DOCUMENTATION=OFF \
          -DVTK_BUILD_EXAMPLES=OFF \
          -DVTK_DATA_EXCLUDE_FROM_ALL:BOOL=ON \
          -DVTK_MODULE_ENABLE_VTK_PythonInterpreter:STRING=NO \
          -DVTK_MODULE_ENABLE_VTK_WebCore:STRING=YES \
          -DVTK_MODULE_ENABLE_VTK_WebGLExporter:STRING=YES \
          -DVTK_MODULE_ENABLE_VTK_WebPython:STRING=YES \
          -DVTK_WHEEL_BUILD=ON \
          -DVTK_PYTHON_VERSION=3 \
          -DVTK_WRAP_PYTHON=ON \
          -DVTK_OPENGL_HAS_EGL=False \
          -DPython3_EXECUTABLE=$PYBIN ../
    ninja-build

    # build wheel
    rm -rf dist
    $PYBIN setup.py bdist_wheel

    # cleanup wheel
    rm -rf wheelhouse
    auditwheel repair dist/*.whl
    cp wheelhouse/vtk*.whl /io/wheels

Be sure to either enable or disable ``DVTK_OPENGL_HAS_EGL`` depending
on if you want ``EGL`` enabled for your wheel.
