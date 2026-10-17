.. _cli_build:

pyvista build
-------------
Command-line interface for building a VTK backend wheel from source.

``pyvista build vtk`` clones VTK, checks for the build tools, configures and
compiles it with the options PyVista's CI uses, and writes a wheel. It works
on Linux, macOS, and Windows. See :ref:`building_vtk` for the eight steps in
detail, the module groups, portable Linux wheels, and the developer loop.

.. versionadded:: 0.50

Examples
********
Build VTK with the default tools, source and configuration, printing every
command as it runs:

.. code-block:: bash

    pyvista build vtk

Walk through the build one step at a time, confirming each step before any
command runs:

.. code-block:: bash

    pyvista build vtk --interactive

Build a different backend, such as the `cvista <https://github.com/pyvista/cvista>`_
fork (see :ref:`vtk_backend_install`), or your own fork of VTK:

.. code-block:: bash

    pyvista build cvista
    pyvista build myfork --url https://github.com/me/vtk.git --ref my-branch

Build VTK's ``master`` branch, or any tag or commit:

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

Print the commands a build would run without running any of them. With
``--interactive`` each question is shown with the answer it would take, which
is the quickest way to see what the eight steps do:

.. code-block:: bash

    pyvista build vtk --interactive --dry-run

.. command-output:: COLUMNS=100 pyvista build vtk --interactive --dry-run --build-dir vtk-build
   :shell:

API Reference
*************
Show the output from ``pyvista build --help``.

.. command-output:: pyvista build --help
