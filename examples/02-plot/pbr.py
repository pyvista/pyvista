"""
.. _pbr_example:

Physically Based Rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~

VTK 9 introduced Physically Based Rendering (PBR) and we have exposed
that functionality in PyVista. Read the `blog about PBR
<https://blog.kitware.com/vtk-pbr/>`_ for more details.

PBR is only supported for :class:`pyvista.PolyData` and can be
triggered via the ``pbr`` keyword argument of ``add_mesh``. Also use
the ``metallic`` and ``roughness`` arguments for further control.

Let's show off this functionality by rendering a high quality mesh of
a statue as though it were metallic.

"""

# sphinx_gallery_start_ignore
# physically based rendering does not seem to work in vtk-js
from __future__ import annotations

PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

from itertools import product

import pyvista as pv
from pyvista import examples

# Load the statue mesh
mesh = examples.download_nefertiti()
mesh.rotate_x(-90.0, inplace=True)  # rotate to orient with the skybox

# Download skybox
cubemap = examples.download_sky_box_cube_map()


# %%
# Let's render the mesh with a base color of "linen" to give it a metal looking
# finish.
pl = pv.Plotter()
pl.add_actor(cubemap.to_skybox())
pl.set_environment_texture(cubemap)  # For reflecting the environment off the mesh
pl.add_mesh(mesh, color='linen', pbr=True, metallic=0.8, roughness=0.1, diffuse=1)

# Define a nice camera perspective
cpos = pv.CameraPosition(
    position=(-313.40, 66.09, 1000.61), focal_point=(0.0, 0.0, 0.0), viewup=(0.018, 0.99, -0.06)
)

pl.show(cpos=cpos)


# %%
# Show the variation of the metallic and roughness parameters.
#
# Plot with metallic increasing from left to right and roughness
# increasing from bottom to top.

colors = ['red', 'teal', 'black', 'orange', 'silver']

pl = pv.Plotter()
pl.set_environment_texture(cubemap)

for i, j in product(range(5), range(6)):
    sphere = pv.Sphere(radius=0.5, center=(0.0, 4 - i, j))
    pl.add_mesh(sphere, color=colors[i], pbr=True, metallic=i / 4, roughness=j / 5)

pl.view_vector((-1, 0, 0), (0, 1, 0))
pl.show()


# %%
# Combine custom lighting and physically based rendering.

# download louis model
mesh = examples.download_louis_louvre()
mesh.rotate_z(140, inplace=True)


pl = pv.Plotter(lighting=None)
pl.set_background('black')
pl.add_mesh(mesh, color='linen', pbr=True, metallic=0.5, roughness=0.5, diffuse=1)


# set up lighting
light = pv.Light(position=(-2, 2, 0), focal_point=(0, 0, 0), color='white')
pl.add_light(light)

light = pv.Light(position=(2, 0, 0), focal_point=(0, 0, 0), color=(0.7, 0.0862, 0.0549))
pl.add_light(light)

light = pv.Light(position=(0, 0, 10), focal_point=(0, 0, 0), color='white')
pl.add_light(light)


# plot with a good camera position
pl.camera_position = pv.CameraPosition(
    position=(9.51, 13.92, 15.81), focal_point=(-2.836, -0.93, 10.2), viewup=(-0.22, -0.18, 0.959)
)
cpos = pl.show()
# %%
# .. tags:: plot
