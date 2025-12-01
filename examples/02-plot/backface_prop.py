"""
.. _backface_prop_example:

Setting Backface Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default front and backface rendering uses the same properties. In certain
situations it can be useful to set different properties for backfaces than
for frontfaces.

One straightforward example is when a closed (or close enough) surface has a
different color on the inside. Note that the notion of "inside" and "outside"
depend on the orientation of the surface normals:
"""

# sphinx_gallery_thumbnail_number = 1

# sphinx_gallery_start_ignore
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "pyvista",
# ]
# ///

# backface properties do not work in interactive
from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

import numpy as np

import pyvista as pv
from pyvista import examples

mesh = pv.ParametricEllipsoid(min_v=np.pi / 2 - 0.2, max_v=np.pi / 2 + 0.2)

# create a shifted copy with flipped normals
mesh_inside_out = mesh.translate((0, 0, 1), inplace=False)
mesh_inside_out.compute_normals(flip_normals=True, inplace=True)
meshes = mesh + mesh_inside_out

backface_params = dict(color='orangered')
meshes.plot(color='aquamarine', backface_params=backface_params, smooth_shading=True)


# %%
# A more interesting use case is helping visualize the orientation of complex,
# self-intersecting surfaces. For instance :func:`Catalan's minimal surface
# <pyvista.ParametricCatalanMinimal>` has a complex shape, and coloring the
# front and backfaces differently helps viewers comprehend the intricate
# structure of the surface. This example also demonstrates use of the
# :attr:`backface_prop <pyvista.Actor.backface_prop>` property of the
# :class:`pyvista.Actor` class.

catalan = pv.ParametricCatalanMinimal()
pl = pv.Plotter()
actor = pl.add_mesh(catalan, color='dodgerblue', smooth_shading=True)
bprop = actor.backface_prop
bprop.color = 'forestgreen'
bprop.specular = 1.0
bprop.specular_power = 50.0
pl.show()


# %%
# In the case of non-orientable surfaces, adding specific backface properties can
# make the non-orientable quality very obvious by the emergence of "seams"
# where the face properties are discontinuous.

henneberg = pv.ParametricHenneberg().scale(0.25, inplace=False)
klein = pv.ParametricKlein().rotate_z(150, inplace=False).translate((6, 0, 0), inplace=False)
meshes = henneberg + klein

backface_params = dict(color='mediumseagreen', specular=1.0, specular_power=50.0)
meshes.plot(color='gold', backface_params=backface_params, smooth_shading=True)


# %%
# Of course we aren't constrained to only setting distinct colors for backfaces;
# most :class:`pyvista.Property` attributes can be overridden. However, some of
# these have no effect, while others merely don't make any sense. For instance,
# most objects have the same opacity no matter which direction you look at them.
# Here is a GIF animation circling around such an asymmetrically opaque Möbius
# strip:

mobius = pv.ParametricMobius().rotate_z(-90, inplace=False)
backface_params = dict(opacity=0.5)
pl = pv.Plotter()
pl.add_mesh(mobius, color='deepskyblue', backface_params=backface_params, smooth_shading=True)
pl.open_gif('mobius_semiopaque.gif')

viewup = [0, 0, 1]
orbit = pl.generate_orbital_path(n_points=24, shift=0.0, viewup=viewup)
pl.orbit_on_path(orbit, write_frames=True, viewup=viewup, step=0.02)


# %%
# Apply Backface Properties to Textured Meshes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Backface textures can also be applied to meshes that have textures applied to
# them. For this example we load the globe texture with
# :func:`pyvista.examples.load_globe() <pyvista.examples.examples.load_globe>`,
# clip it, and then apply a different color to the interior surface.
#
# The lighting has been disabled for this example to demonstrate how you can
# make the interior of the surface appear occluded without any directional
# lighting simply by providing a different color for backface.

globe = examples.load_globe()
texture = examples.load_globe_texture()
clipped = globe.clip(normal='z', value=4.37e9)

pl = pv.Plotter()
pl.add_mesh(
    clipped,
    backface_params={'color': [0.2, 0.2, 0.2]},
    lighting=False,
    smooth_shading=True,
    texture=texture,
)
pl.show()


# %%
# Backface Properties and Physically Based Rendering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note that backfaces are automatically culled when physically based rendering
# is enabled, regardless of the settings of backface parameters.

sphere = pv.Sphere()
clipped_sphere = sphere.clip(normal='z', value=0.4)

pl = pv.Plotter()
pl.set_environment_texture(examples.download_sky_box_cube_map())
pl.add_mesh(
    clipped_sphere,
    backface_params={'color': 'r'},
    pbr=True,
    metallic=1.0,
    roughness=0.2,
)
pl.show()


# %%
# See also the :ref:`sphere_eversion_example` example which relies on
# distinguishing the inside and the outside of a sphere.
#
# .. tags:: plot
