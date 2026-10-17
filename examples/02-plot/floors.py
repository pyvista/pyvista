"""
.. _floors_example:

Plot With Floors
~~~~~~~~~~~~~~~~

Add a floor/wall at the scene boundary using :func:`~pyvista.Plotter.add_floor`.

"""

import pyvista as pv
from pyvista import examples

mesh = examples.download_bunny()

pl = pv.Plotter()
pl.add_mesh(mesh)
pl.add_floor('-y')
pl.add_floor('-z')
pl.show()
# %%
# .. tags:: plot
