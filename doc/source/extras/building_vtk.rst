.. _building_vtk:

Building VTK
============
Kitware publishes VTK wheels on `PyPI <https://pypi.org/project/vtk/>`_,
and PyVista installs those by default. Building VTK from source is for the
cases they do not cover: a new Python release, a VTK branch or commit, GPU
off-screen rendering, or a module the published wheel leaves out.

`pyvista-build <https://pyvista.github.io/pyvista-build/>`_ does that with
one command on Linux, macOS, and Windows, using the options PyVista's own CI
uses:

.. code-block:: bash

    pip install pyvista-build
    pyvista-build vtk

The wheel is built for, and installed into, the active virtual environment.
Once installed, PyVista picks it up like the stock one. Its documentation
covers:

- `The steps <https://pyvista.github.io/pyvista-build/building.html#the-steps>`_
  of a build, and the flags that change each one.
- `Scripted use <https://pyvista.github.io/pyvista-build/building.html#scripted-use>`_
  and the `developer loop
  <https://pyvista.github.io/pyvista-build/building.html#developer-loop>`_ of
  editing VTK and rebuilding.
- `Portable Linux wheels
  <https://pyvista.github.io/pyvista-build/building.html#portable-linux-wheels>`_
  built inside a manylinux container.
- `CI workflows
  <https://pyvista.github.io/pyvista-build/building.html#ci-workflows>`_
  with a compiler cache.
- The `command reference <https://pyvista.github.io/pyvista-build/cli.html>`_.

PyVista's nightly `VTK master build
<https://github.com/pyvista/pyvista/blob/main/.github/workflows/vtk-master-wheels.yml>`_
runs the same command against VTK's master branch on all three operating
systems and tests PyVista against the result.

.. _gpu_off_screen:

Off-Screen Plotting GPU Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The stock Linux wheels already render off-screen without an X server: VTK
picks EGL for GPU rendering, or OSMesa for software rendering, at run time.
To build a wheel that does not link against X at all:

.. code-block:: bash

    pyvista-build vtk --with headless

Such a wheel cannot open on-screen windows, so use it only where you save
static images or stream the render window to another computer with a
display (for example, using ``pyvista.set_jupyter_backend('server')`` and
JupyterLab). See `Off-Screen Plotting GPU Support
<https://pyvista.github.io/pyvista-build/building.html#off-screen-plotting-gpu-support>`_.
