.. _cli_compare:

pyvista compare
---------------
Command-line interface for comparing two or more mesh files side-by-side.

Using ``pyvista compare`` is similar to calling :func:`~pyvista.plot_compare` in Python.

Examples
********

.. note::
    To run the examples yourself locally, first change directory to ``pyvista/examples``, e.g.

    .. code-block:: bash

        cd $(python -c "import pyvista.examples, pathlib; print(pathlib.Path(pyvista.examples.__file__).parent)")

Compare two mesh files. Each is drawn in its own subplot and labeled with its filename.

.. command-output:: pyvista compare ant.ply nut.ply
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import pyvista as pv
   from pathlib import Path
   examples = Path('source') / '_local_examples'
   pv.plot_compare({p.stem: pv.read(p) for p in [examples / 'ant.ply', examples / 'nut.ply']})

Compare any number of files using wildcard patterns. The subplots are arranged in a
compact grid which is never taller than it is wide.

.. command-output:: pyvista compare *.ply
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import pyvista as pv
   from pathlib import Path
   examples = Path('source') / '_local_examples'
   pv.plot_compare({p.stem: pv.read(p) for p in sorted(examples.glob('*.ply'))})

Use ``--outline`` to draw the bounds of every mesh in each subplot, which gives the
comparison a common frame of reference and shows where each mesh sits within it.

.. command-output:: pyvista compare ant.ply nut.ply --outline
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import pyvista as pv
   from pathlib import Path
   examples = Path('source') / '_local_examples'
   meshes = {p.stem: pv.read(p) for p in [examples / 'ant.ply', examples / 'nut.ply']}
   outline = pv.MultiBlock(list(meshes.values())).outline()
   pv.plot_compare(meshes, reference_mesh=outline)

Each subplot is framed on its own mesh unless the meshes are of a comparable size, in
which case they share a single camera. Use ``--link`` to share one in any case, which
shows the meshes at their true relative size. The airplane is some forty times the size
of the ant, so the ant is barely visible beside it.

.. command-output:: pyvista compare airplane.ply ant.ply --link
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import warnings

   import pyvista as pv
   from pathlib import Path
   examples = Path('source') / '_local_examples'
   meshes = {p.stem: pv.read(p) for p in [examples / 'airplane.ply', examples / 'ant.ply']}
   with warnings.catch_warnings():
       warnings.simplefilter('ignore')
       pv.plot_compare(meshes, link=True)

Use ``--normalize`` to resize every mesh to the same size instead, and compare their
shapes rather than their sizes. The files themselves are left as they are.

.. command-output:: pyvista compare airplane.ply ant.ply --normalize
   :extraargs: --off-screen
   :cwd: /_local_examples

.. pyvista-plot::
   :include-source: False

   import pyvista as pv
   from pathlib import Path
   examples = Path('source') / '_local_examples'
   meshes = {p.stem: pv.read(p) for p in [examples / 'airplane.ply', examples / 'ant.ply']}
   pv.plot_compare(meshes, normalize=True)

Compare mesh files off-screen and save a screenshot.

.. command-output:: pyvista compare *.ply --screenshot output.png --off-screen
   :cwd: /_local_examples

API Reference
*************
Show the output from ``pyvista compare --help``.

.. command-output:: pyvista compare --help
