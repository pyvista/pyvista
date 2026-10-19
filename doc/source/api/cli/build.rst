.. _cli_build:

pyvista build
-------------
Command-line interface for building a VTK backend wheel from source.

``pyvista build vtk`` clones VTK, checks for the build tools, configures and
compiles it much like PyVista's CI does, and writes a wheel. It works
on Linux, macOS, and Windows. See :ref:`building_vtk` for the eight steps in
detail, the module groups, portable Linux wheels, and the developer loop.

.. versionadded:: 0.50

Examples
********
Build VTK with the default tools, source, and configuration, printing every
command as it runs:

.. code-block:: bash

    pyvista build vtk

Walk through the build one step at a time, confirming each step before any
command runs:

.. code-block:: bash

    pyvista build vtk --interactive

Build a different backend, such as the `cvista <https://github.com/pyvista/cvista>`_
fork (see :ref:`vtk_backend`), or your own fork of VTK in its own build
directory:

.. code-block:: bash

    pyvista build cvista
    pyvista build vtk --url https://github.com/me/vtk.git --ref my-branch --build-dir vtk-fork-build

Build without being asked, for scripts and CI. The list of what gets
installed is still printed:

.. code-block:: bash

    pyvista build vtk --yes

Build VTK's ``master`` branch, or any tag, branch, or commit:

.. code-block:: bash

    pyvista build vtk --ref master

Customize the module groups, for example a wheel without the web export
modules, or a minimal one with rendering only:

.. code-block:: bash

    pyvista build vtk --without web
    pyvista build vtk --minimal --with rendering

Customize the build itself: the build type, a CMake cache entry passed
straight through, and the number of compile jobs:

.. code-block:: bash

    pyvista build vtk --build-type Debug -D VTK_ENABLE_LOGGING=OFF --jobs 8

Install the wheel into the current environment when the build finishes:

.. code-block:: bash

    pyvista build vtk --install

Build a portable Linux wheel inside a manylinux container (needs Docker, see
:ref:`building_vtk`):

.. code-block:: bash

    pyvista build vtk --manylinux

Print the commands a build would run without running any of them, which is the
quickest way to see what the eight steps do. For ``cvista``, whose manifest lives
in its source tree, a dry run needs an existing checkout:

.. code-block:: bash

    pyvista build vtk --dry-run

.. program-output:: COLUMNS=90 pyvista build vtk --dry-run
   :shell:

API Reference
*************
Show the output from ``pyvista build --help``.

.. command-output:: pyvista build --help
