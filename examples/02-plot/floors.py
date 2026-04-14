"""
.. _floors_example:

Plot with Floors
~~~~~~~~~~~~~~~~

Add a floor/wall at the boundary of the rendering scene
using :func:`~pyvista.Plotter.add_floor`.
"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

mesh = examples.download_dragon()

pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_floor('-y')
pl.add_floor('-z')
pl.show()
# %%
# .. tags:: plot
