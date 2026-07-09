.. _cli_plot:

pyvista plot
------------
Command-line interface for plotting one or more mesh files in an interactive window.

Using ``pyvista plot`` is similar to calling :func:`~pyvista.plot` in Python.

Examples
********
Plot a single mesh file.

.. command-output:: pyvista plot ant.ply
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import pyvista as pv
   from pathlib import Path
   pv.plot(Path('source') / '_local_examples' / 'ant.ply')

Plot multiple PLY mesh files in a single window using wildcard patterns and use ``--zoom``.

.. command-output:: pyvista plot *.ply --zoom 2
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import pyvista as pv
   from pathlib import Path
   pv.plot(list((Path('source') / '_local_examples').glob('*.ply')), zoom=2)

Plot a mesh file off-screen and save a screenshot.

.. command-output:: pyvista plot ant.ply --screenshot output.png --off-screen
   :cwd: /_local_examples

API Reference
*************
Show the output from ``pyvista plot --help``.

.. command-output:: pyvista plot --help
