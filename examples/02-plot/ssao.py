"""

.. _ssao_example:

Demonstrate the usage of surface space ambient occlusion.

Surface space ambient occlusion (SSAO) can approximate shadows more
efficiently than ray-tracing and produce similar results. Use this when you wish
to plot the occlusion effect that nearby meshes have on each other by blocking
nearby light sources.

See `Kitware: Screen-Space Ambient Occlusion
<https://www.kitware.com/ssao/>`_ for more details

"""

# First, let's create several spheres tightly touching each other.

import pyvista as pv
from pyvista import examples

grid = pv.UniformGrid(dims=(4, 4, 4))
spheres = grid.glyph(geom=pv.Sphere(), orient=False, scale=False)
spheres

###############################################################################
# Plot with defaults
# ~~~~~~~~~~~~~~~~~~
# Let's plot this without SSAO. Note how the lighting is identical for each
# sphere.

spheres.plot(smooth_shading=True)


###############################################################################
# Plot with SSAO
# ~~~~~~~~~~~~~~
# Now plot this with SSAO. Note how adjacent spheres affect the lighting of each
# other.
#
# We've also enabled SSAA anti-aliasing to smooth out some visual artifacts
# that occur with SSAO.

pl = pv.Plotter()
pl.add_mesh(spheres, smooth_shading=True)
pl.enable_ssao()
pl.enable_anti_aliasing('ssaa')
pl.show()


###############################################################################
# Plot a carburetor without SSAO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here's another example without SSAO. This is a surface scan of a carburetor.

mesh = examples.download_carburetor()

pl = pv.Plotter()
pl.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True)
pl.camera_position = 'xy'
pl.camera.zoom(2)
pl.show()


###############################################################################
# Plot a carburetor with SSAO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here's the same carburetor with SSAO. Note how we had to increase both
# ``radius`` and ``bias`` due to the relative scene size.

pl = pv.Plotter()
pl.add_mesh(mesh, smooth_shading=True, split_sharp_edges=True)
pl.enable_ssao(radius=10, bias=1)
pl.camera_position = 'xy'
pl.camera.zoom(2)
pl.show()
