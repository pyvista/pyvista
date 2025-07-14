"""
.. _perlin_noise_3d_example:

Sample Function: Perlin Noise in 3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we use :func:`pyvista.core.utilities.features.sample_function` to sample
Perlin noise over a region to generate random terrain.

Video games like Minecraft use Perlin noise to create terrain.  Here,
we create a voxelized mesh similar to a Minecraft "cave".

"""

from __future__ import annotations

import pyvista as pv

# %%
# Generate Perlin Noise over a 3D StructuredGrid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Feel free to change the values of ``freq`` to change the shape of
# the "caves".  For example, lowering the frequency will make the
# caves larger and more expansive, while a higher frequency in any
# direction will make the caves appear more "vein-like" and less open.
#
# Change the threshold to reduce or increase the percent of the
# terrain that is open or closed

freq = (1, 1, 1)
noise = pv.perlin_noise(1, freq, (0, 0, 0))
grid = pv.sample_function(noise, bounds=[0, 3.0, -0, 1.0, 0, 1.0], dim=(120, 40, 40))
out = grid.threshold(0.02)
out

# %%
# color limits without blue
mn, mx = [out['scalars'].min(), out['scalars'].max()]
clim = (mn, mx * 1.8)

out.plot(
    cmap='gist_earth_r',
    background='white',
    show_scalar_bar=False,
    lighting=True,
    clim=clim,
    show_edges=False,
)
# %%
# .. tags:: filter
