.. _building_vtk:

Building VTK
============
Kitware provides Python wheels for VTK at `PyPI VTK
<https://pypi.org/project/vtk/>`_, but there are situations where you
may need to build VTK from source: a new release of Python, EGL or
OSMesa rendering, modules the stock wheel leaves out, a VTK branch with
a fix you need, or a backend such as `cvista
<https://github.com/pyvista/cvista>`_.

PyVista ships a command that does this for you:

.. code-block:: bash

    pyvista build vtk

.. versionadded:: 0.50

It works on Linux, macOS, and Windows and follows the official `Building
VTK <https://docs.vtk.org/en/latest/build_instructions/index.html>`_
instructions, with the options PyVista's own CI uses. Run from a
terminal, it walks through the build one step at a time, says what it
is about to do and where, and waits for a yes. Press Enter to accept
the default at each step.

The Steps
~~~~~~~~~
1. **Source.** VTK is cloned into ``./vtk-build/src`` at the release tag
   PyVista supports. Use ``--ref`` for another tag, branch or commit, and
   ``--build-dir`` to put everything somewhere else. An existing checkout
   is reused as is, local edits included.
2. **Tools.** ``git``, ``cmake``, ``ninja``, a C++ compiler and the wheel
   repair tool are checked. ``cmake``, ``ninja`` and the repair tool can
   be installed into the current environment with ``pip`` after you
   confirm; a missing compiler stops with instructions for your
   operating system. On Windows, Visual Studio's build environment is
   imported automatically and NASM is downloaded with VTK's own script.
   A compiler cache (``ccache`` or ``sccache``) is used when one is on
   ``PATH``, so a second build after editing the source is fast.
3. **Build options.** The modules to build are organized into named
   groups (``rendering``, ``io``, ``web``, ``headless-egl``, ...), each
   with a one-line description. The default set matches the stock wheel
   plus what PyVista's CI enables. At the prompt, ``n`` asks about each
   group in turn, and the command-line flags work there as well:

   .. code-block:: bash

       pyvista build vtk --without web,matplotlib
       pyvista build vtk --minimal --with rendering
       pyvista build vtk --all
       pyvista build vtk --opt smp=TBB

4. **Configure** shows the build type, compiler cache, version suffix
   and extra CMake cache entries. ``n`` asks about each in turn, and
   ``--build-type``, ``--compiler-cache``, ``--version-suffix`` and
   ``-D`` change one directly. Then ``cmake`` runs. Every command is
   printed in full before it runs.
5. **Build** runs ``ninja`` and logs the full output to ``build.log``.
6. **Wheel** runs ``setup.py bdist_wheel`` in the build tree.
7. **Repair** copies VTK's shared libraries into the wheel so it installs
   anywhere, not only next to the build tree. Skip it with
   ``--no-repair``.
8. **Install** offers to install the wheel into the current environment.
   ``--install`` says yes up front; otherwise the wheel's path and the
   ``pip`` command are printed.

Non-Interactive Use
~~~~~~~~~~~~~~~~~~~
``--yes`` accepts every default and ``--dry-run`` prints the commands
without running anything. When the command is not attached to a
terminal, for example in CI, it behaves as if ``--yes`` were given.
Optional steps keep their own switches, so ``--yes --no-repair`` skips
the repair step.

.. code-block:: bash

    pyvista build vtk --yes --ref master --build-type RelWithDebInfo --jobs 8

Developer Loop
~~~~~~~~~~~~~~
The build tree is kept between runs. Edit files under ``vtk-build/src``
and run ``pyvista build vtk`` again: configure is skipped unless the
options changed, ``ninja`` rebuilds what the edit touched and a new
wheel is written. ``--reconfigure`` forces ``cmake`` to run again,
``--clean`` deletes the build tree first, ``--update`` fetches and
checks out ``--ref``, and ``-D KEY=VALUE`` passes any CMake cache
entry straight through:

.. code-block:: bash

    pyvista build vtk --build-type Debug -D VTK_ENABLE_LOGGING=OFF

Other Backends
~~~~~~~~~~~~~~
The same command builds any VTK backend whose source tree carries a
``pyvista-build.toml`` manifest. That file declares the groups and
options the backend offers, its tool requirements, per-platform CMake
init caches and the extra commands it needs; stock VTK's manifest ships
with PyVista. ``cvista`` is known by name, and a fork is reachable by
URL:

.. code-block:: bash

    pyvista build cvista
    pyvista build myfork --url https://github.com/me/vtk.git

.. _gpu_off_screen:

Off-Screen Plotting GPU Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VTK supports rendering with EGL, enabling rapid off-screen rendering
using GPU hardware acceleration without installing a virtual
framebuffer. The default VTK wheels are not built with this feature. On
Linux, build a wheel with it enabled with:

.. code-block:: bash

    sudo apt install libegl1-mesa-dev  # or: dnf install mesa-libEGL-devel
    pyvista build vtk --with headless-egl

This disables any plotting using the X server, so be prepared to use
this module only on a headless display where you either intend to save
static images or stream the render window to another computer with a
display (for example, using ``pyvista.set_jupyter_backend('server')`` and
JupyterLab). In other words, this wheel will make VTK unusable outside
of an off-screen environment. ``--with headless-osmesa`` builds the
software-rendering equivalent instead.

Portable Linux Wheels
~~~~~~~~~~~~~~~~~~~~~
A wheel built on Linux is tagged for the ``glibc`` of the machine that
built it and may not install elsewhere. To share one, run the command
inside a `manylinux <https://github.com/pypa/manylinux>`_ container, as
PyVista's own CI does:

.. code-block:: bash

    docker run --rm -it -v "$PWD:/work" -w /work quay.io/pypa/manylinux2014_x86_64 bash
    yum install -y libX11-devel libXcursor-devel mesa-libGL-devel
    /opt/python/cp314-cp314/bin/python -m pip install pyvista
    /opt/python/cp314-cp314/bin/pyvista build vtk --yes

The repaired wheel is written to ``vtk-build/wheelhouse``.

Building by Hand
~~~~~~~~~~~~~~~~
``pyvista build vtk --dry-run`` prints the exact ``cmake`` and
``setup.py`` commands, which is the quickest way to get a starting point
for a manual build. The essential configuration is:

.. code-block:: bash

    git clone https://gitlab.kitware.com/vtk/vtk.git
    cmake -S vtk -B vtk/build -GNinja \
          -DCMAKE_BUILD_TYPE=Release \
          -DVTK_WHEEL_BUILD=ON \
          -DVTK_WRAP_PYTHON=ON \
          -DVTK_BUILD_TESTING=OFF \
          -DVTK_BUILD_DOCUMENTATION=OFF \
          -DVTK_BUILD_EXAMPLES=OFF \
          -DPython3_EXECUTABLE=$(which python)
    cmake --build vtk/build --parallel
    cd vtk/build && python setup.py bdist_wheel

The wheel is written to ``vtk/build/dist``.
