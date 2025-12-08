"""
.. _warp_by_vector_example:

Warping by Vectors
~~~~~~~~~~~~~~~~~~

This example applies the :meth:`~pyvista.DataSetFilters.warp_by_vector`
filter to a sphere mesh that has 3D displacement vectors defined at each node.
"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

# %%
# We first compare the unwarped sphere to the warped sphere.
from __future__ import annotations

from itertools import product

import pyvista as pv
from pyvista import examples

sphere = examples.load_sphere_vectors()
warped = sphere.warp_by_vector()

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.add_text('Before warp')
pl.add_mesh(sphere, color='white')
pl.subplot(0, 1)
pl.add_text('After warp')
pl.add_mesh(warped, color='white')
pl.show()

# %%
# We then use several values for the scale factor applied to the warp
# operation. Applying a warping factor that is too high can often lead to
# unrealistic results.

warp_factors = [0, 1.5, 3.5, 5.5]
pl = pv.Plotter(shape=(2, 2))
for i, j in product(range(2), repeat=2):
    idx = 2 * i + j
    pl.subplot(i, j)
    pl.add_mesh(sphere.warp_by_vector(factor=warp_factors[idx]))
    pl.add_text(f'factor={warp_factors[idx]}')
pl.show()
# %%
# .. tags:: filter
