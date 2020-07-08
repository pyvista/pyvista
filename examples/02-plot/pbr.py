"""
Physically Based Rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~

VTK 9 introduced Physically Based Rendering (PBR) and we have exposed that
functionality in PyVista. Read the `blog about PBR <https://blog.kitware.com/vtk-pbr/>`_
for more details.

PBR is only supported for :class:`pyvista.PolyData` and can be triggered via
the ``pbr`` keyword argument of ``add_mesh``. Also use the ``metallic`` and ``roughness` arguments for furhter control.

Let's show of this functionality by rendering a high quality mesh of a statue
as though it were metallic.
"""

import pyvista as pv
from pyvista import examples

# Load the statue mesh
mesh = examples.download_nefertiti()
mesh.rotate_x(-90.) # rotate to orient with the skybox

# Download skybox
texture = examples.download_sky_box_cube_map()
skybox = texture.to_skybox()

# Define a nice camera perspective
cpos = [(-313.40, 66.09, 1000.61),
        (0.0, 0.0, 0.0),
        (0.018, 0.99, -0.06)]

###############################################################################
# Let's render the mesh with a base color of "linen" to give it a metal looking
# finish.
p = pv.Plotter()
p.add_actor(skybox)
p.set_environment_texture(texture) # For refecting the environment off the mesh
p.add_mesh(mesh, color='linen',
           pbr=True, metallic=0.8, roughness=.1,
           diffuse=1)
p.show(cpos=cpos)
