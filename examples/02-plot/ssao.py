"""

.. _ssao_example:

Surface Space Ambient Occlusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrate the usage of surface space ambient occlusion.

Surface space ambient occlusion (SSAO) can approximate shadows more
efficiently than ray-tracing and produce similar results. Use this when you wish
to plot the occlusion effect that nearby meshes have on each other by blocking
nearby light sources.

See `Kitware: Screen-Space Ambient Occlusion
<https://www.kitware.com/ssao/>`_ for more details

"""

# First, let's create several spheres tightly touching each other.
import numpy as np

import pyvista as pv
from pyvista import examples

grid = pv.UniformGrid(dims=(4, 4, 4))
spheres = grid.glyph(geom=pv.Sphere(), orient=False, scale=False)
spheres

###############################################################################
# Convert Position to RBG Colors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate some fun colors to plot our spheres with.
#
colors = spheres.points
colors -= colors.min(axis=0)
colors /= colors.max(axis=0)
colors = (colors * 255).astype(np.uint8)


###############################################################################
# Plot with defaults
# ~~~~~~~~~~~~~~~~~~
# Let's plot this without SSAO. Note how the lighting is identical for each
# sphere.

# always include an environment texture when using physically based rendering.
cubemap = examples.download_cubemap_park()

pl = pv.Plotter()
pl.add_mesh(spheres, scalars=colors, rgb=True, smooth_shading=True, pbr=True, metallic=1.0)
pl.set_environment_texture(cubemap)
pl.enable_anti_aliasing('ssaa')
pl.show()


###############################################################################
# Plot with SSAO
# ~~~~~~~~~~~~~~
# Now plot this with SSAO. Note how adjacent spheres affect the lighting of
# each other to make it look less artificial.
#
# We've also enabled SSAA anti-aliasing to smooth out some visual artifacts
# that occur with SSAO. We've also increased the ``kernel_size`` to improve the
# quality of the SSAO.

pl = pv.Plotter()
pl.add_mesh(spheres, scalars=colors, rgb=True, smooth_shading=True, pbr=True, metallic=1.0)
pl.set_environment_texture(cubemap)
pl.enable_ssao(kernel_size=2048)
pl.enable_anti_aliasing('ssaa')
pl.show()


###############################################################################
# Plot a carburetor without SSAO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here's another example without SSAO. This is a simple CAD model.

mesh = examples.download_cad_model()

pl = pv.Plotter()
pl.add_mesh(
    mesh, smooth_shading=True, split_sharp_edges=True, pbr=True, metallic=0.5, roughness=0.5
)
pl.set_environment_texture(cubemap)
pl.enable_anti_aliasing('ssaa')
pl.camera.zoom(1.7)
pl.show()


###############################################################################
# Plot a carburetor with SSAO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here's the same CAD model with SSAO. Note how we had to increase both
# ``radius`` and ``bias`` due to the relative scene size.

pl = pv.Plotter()
pl.add_mesh(
    mesh, smooth_shading=True, split_sharp_edges=True, pbr=True, metallic=0.5, roughness=0.5
)
pl.enable_ssao(radius=2, bias=1)
pl.set_environment_texture(cubemap)
pl.enable_anti_aliasing('ssaa')
pl.camera.zoom(1.7)
pl.show()
