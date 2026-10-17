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
instructions, with the options PyVista's own CI uses. By default it
builds with those defaults straight through, printing every command as
it runs. Add ``--interactive`` to walk through the build one step at a
time: each step shows its settings, says what it is about to do and
where, and waits for a yes. Press Enter to accept the default, answer
``n`` to go through the settings one by one, or type the same flags the
command line takes to change one in place.

.. code-block:: bash

    pyvista build vtk --interactive

The command reference with more examples is at :ref:`cli_build`. The
result is a wheel, so the build selects itself through the usual
mechanism once installed: see :envvar:`PYVISTA_VTK_BACKEND` and
:ref:`the backend section of the installation guide <vtk_backend_install>`.

The Steps
~~~~~~~~~
1. **Source.** VTK is cloned into ``./vtk-build/src`` at the release tag
   PyVista supports. Use ``--ref`` for another tag, branch or commit, and
   ``--build-dir`` to put everything somewhere else. An existing checkout
   is reused as is, local edits included.
2. **Tools.** ``git``, ``cmake``, ``ninja``, a C++ compiler and the wheel
   repair tool are checked (see :ref:`build_tooling` for what each one
   does). ``cmake``, ``ninja`` and the repair tool can be installed into
   the current environment with ``pip`` after you confirm; a missing
   compiler stops with instructions for your operating system. On
   Windows, Visual Studio's build environment is imported automatically
   and NASM is downloaded with VTK's own script. A compiler cache
   (``ccache`` or ``sccache``) is used when one is on ``PATH``, so a
   second build after editing the source is fast.
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

Scripted Use
~~~~~~~~~~~~
Without ``--interactive`` nothing is asked, so the command is ready for
CI and scripts as is. ``--dry-run`` prints the commands without running
anything, and with ``--interactive`` it also prints each question with
the answer it would take. Optional steps keep their own switches, so
``--no-repair`` skips the repair step and ``--install`` installs the
wheel at the end.

.. code-block:: bash

    pyvista build vtk --ref master --build-type RelWithDebInfo --jobs 8
    pyvista build vtk --interactive --dry-run

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
with PyVista. `cvista <https://github.com/pyvista/cvista>`_, the
community fork described in :ref:`vtk_backend_install`, is known by
name, and a fork is reachable by URL:

.. code-block:: bash

    pyvista build cvista
    pyvista build myfork --url https://github.com/me/vtk.git

A backend built this way is selected like any other installed one:
automatically for ``cvista``, or explicitly through
:envvar:`PYVISTA_VTK_BACKEND`.

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
built it and may not install elsewhere. To share one, build it inside a
`manylinux <https://github.com/pypa/manylinux>`_ container, as PyVista's
own CI does. ``--manylinux`` does that through `Docker
<https://docs.docker.com/get-docker/>`_, which must be installed and
running:

.. code-block:: bash

    pyvista build vtk --manylinux

The container image matches your machine's architecture
(``manylinux_2_28_x86_64`` or ``manylinux_2_28_aarch64``) and carries
the same Python version as the one running the command. Inside it the
command installs VTK's build dependencies, PyVista, and ``ccache``, then
runs the eight steps with ``./vtk-build`` mounted as the build tree, so
the repaired wheel lands in ``vtk-build/wheelhouse`` and the compiler
cache is kept between runs. Files written by the container are handed
back to your user. Pass another image with ``--manylinux-image``, and
note that a build tree configured inside the container cannot be
continued outside it. The install step runs on the host afterwards.

.. _build_tooling:

Tooling
~~~~~~~
What the command looks for and offers to install:

- `git <https://git-scm.com/>`_ clones the source and checks out the
  requested tag, branch or commit. It is the one tool the command never
  installs for you.
- `CMake <https://cmake.org/>`_ configures the build: it reads VTK's
  ``CMakeLists.txt`` and the options chosen, finds the compiler and
  Python, and writes the build files. Installed with ``pip`` when
  missing.
- `Ninja <https://ninja-build.org/>`_ is the build tool that CMake
  generates files for; it runs the compiler across all cores and only
  rebuilds what changed. Installed with ``pip`` when missing.
- **A C++ compiler**: GCC or Clang on Linux, Xcode's command-line tools
  on macOS, Visual Studio with the C++ workload on Windows. Never
  installed by the command; a missing one stops with the instructions
  for your operating system.
- `ccache <https://ccache.dev/>`_ and `sccache
  <https://github.com/mozilla/sccache>`_ are compiler caches: they store
  each object file keyed by its inputs, so rebuilding after an edit or a
  reconfigure only recompiles what changed. ``ccache`` is the usual
  choice on Linux and macOS; ``sccache`` is the one that supports MSVC
  on Windows. ``--compiler-cache auto`` uses whichever is on ``PATH``.
- **Wheel repair tools** copy the shared libraries a wheel depends on
  into the wheel itself and set the platform tag, so it installs on a
  machine without the build tree: `auditwheel
  <https://github.com/pypa/auditwheel>`_ on Linux, `delocate
  <https://github.com/matthew-brett/delocate>`_ on macOS and `delvewheel
  <https://github.com/adang1345/delvewheel>`_ on Windows. Installed with
  ``pip`` when missing.
- `NASM <https://www.nasm.us/>`_ is the assembler that VTK's bundled
  libjpeg-turbo uses for its SIMD code paths. Needed on Windows only;
  the command offers to download it with VTK's own script.
- `setuptools <https://setuptools.pypa.io/>`_ and `wheel
  <https://wheel.readthedocs.io/>`_ are the Python packages that
  ``setup.py bdist_wheel`` needs to package the build tree. Installed
  with ``pip`` when missing.
- `Docker <https://docs.docker.com/get-docker/>`_ runs the manylinux
  container for ``--manylinux``. Never installed by the command.

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
