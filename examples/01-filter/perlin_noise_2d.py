"""
.. _perlin_noise_2d_example:

Sample Function: Perlin Noise in 2D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we use :func:`pyvista.core.utilities.features.sample_function` to sample
Perlin noise over a region to generate random terrain.

Perlin noise is atype of gradient noise often used by visual effects
artists to increase the appearance of realism in computer graphics.
Source: `Perlin Noise Wikipedia <https://en.wikipedia.org/wiki/Perlin_noise>`_

The development of Perlin Noise has allowed computer graphics artists
to better represent the complexity of natural phenomena in visual
effects for the motion picture industry.

"""

from __future__ import annotations

import pyvista as pv

# %%
# Generate Perlin Noise over a StructuredGrid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Feel free to change the values of ``freq`` to change the shape of
# the "mountains".  For example, lowering the frequency will make the
# terrain seem more like hills rather than mountains.
freq = [0.689, 0.562, 0.683]
noise = pv.perlin_noise(1, freq, (0, 0, 0))
sampled = pv.sample_function(noise, bounds=(-10, 10, -10, 10, -10, 10), dim=(500, 500, 1))


# %%
# Warp by scalar
# ~~~~~~~~~~~~~~
# Here we warp by scalar to give the terrain some height based on the
# value of the Perlin noise.  This is necessary to the terrain its shape.

mesh = sampled.warp_by_scalar('scalars')
mesh = mesh.extract_surface(algorithm=None)

# clean and smooth a little to reduce Perlin noise artifacts
mesh = mesh.smooth(n_iter=100, inplace=False, relaxation_factor=1)

# This makes the "water" level look flat.
z = mesh.points[:, 2]
diff = z.max() - z.min()

# water level at 70%  (change this to change the water level)
water_percent = 0.7
water_level = z.max() - water_percent * diff
mesh.points[z < water_level, 2] = water_level


# %%
# Show the terrain as a contour plot

# make the water blue
rng = z.max() - z.min()
clim = (z.max() - rng * 1.65, z.max())

pl = pv.Plotter()
pl.add_mesh(
    mesh,
    scalars=z,
    cmap='gist_earth',
    n_colors=10,
    show_scalar_bar=False,
    smooth_shading=True,
    clim=clim,
)
pl.show()


# %%
# Show the terrain with custom lighting and shadows

pl = pv.Plotter(lighting=None)
pl.add_light(
    pv.Light(
        position=(3, 1, 0.5),
        show_actor=True,
        positional=True,
        cone_angle=90,
        intensity=1.2,
    )
)
pl.add_mesh(
    mesh, cmap='gist_earth', show_scalar_bar=False, smooth_shading=True, clim=clim
)
pl.enable_shadows = True
pl.show()
# %%
# .. tags:: filter
