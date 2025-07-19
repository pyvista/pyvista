"""
.. _read_parallel_example:

Parallel Files
~~~~~~~~~~~~~~

The VTK library supports parallel file formats. Reading meshes broken up into
several files is natively supported by VTK and PyVista.
"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

from pathlib import Path

# sphinx_gallery_thumbnail_number = 1
import pyvista as pv
from pyvista import examples

# %%
# Let's go ahead and download the sample dataset containing an
# :class:`pyvista.UnstructuredGrid` broken up into several files.
#
# Let's inspect where this downloaded our dataset by setting ``load=False`` and
# looking at the directory containing the file we downloaded.
filename = examples.download_blood_vessels(load=False)
path = Path(filename).parent
list(path.iterdir())


# %%
list((path / 'T0000000500').iterdir())


# %%
# Note that a ``.pvtu`` file is available alongside a directory. This
# directory contains all the parallel files or pieces that make the whole mesh.
# We can simply read the ``.pvtu`` file and VTK will handle putting the mesh
# together. In PyVista, this is accomplished through :func:`pyvista.read`.
mesh = pv.read(filename)
mesh


# %%
# Plot the pieced together mesh
mesh.plot(scalars='node_value', categories=True)


# %%
mesh.plot(scalars='density')
# %%
# .. tags:: load
