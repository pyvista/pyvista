"""
Physically Based Rendering
~~~~~~~~~~~~~~~~~~~~~~~~~~

VTK 9 introduced Physically Based Rendering (PBR) and we have exposed that
functionality in PyVista. Read the `blog about PBR <https://blog.kitware.com/vtk-pbr/>`_
for more details.

PBR is only supported for :class:`pyvista.PolyData` and can be triggered via
the ``pbr`` keyword argument of ``add_mesh``. Also use the ``metallic`` and ``roughness` arguments for further control.

Let's show off this functionality by rendering a high quality mesh of a statue
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
p.set_environment_texture(texture) # For reflecting the environment off the mesh
p.add_mesh(mesh, color='linen',
           pbr=True, metallic=0.8, roughness=.1,
           diffuse=1)
p.show(cpos=cpos)

###############################################################################

# Helper to create good sphere
Sphere = lambda center: pv.Sphere(center=center, radius=0.5,
                                  theta_resolution=128,
                                  phi_resolution=128).compute_normals()

###############################################################################
colors = ['red', 'teal', 'black', 'orange', 'silver']

p = pv.Plotter()
p.set_environment_texture(texture)

for i in range(5):
    color = colors[i]
    for j in range(6):
        mesh = Sphere((0.0, 4 - i, j))  # flip here for now
        p.add_mesh(mesh, color=color,
                   pbr=True, metallic=i*2/8, roughness=j*2/10,  # i*2/8 for metallic?
                   smooth_shading=True)  # this is the significant new item

p.view_vector((-1,0,0), (0,1,0))
p.show()

###############################################################################
color = 0.7, 0.5, 0.1

p = pv.Plotter()
p.set_environment_texture(texture)
p.add_mesh(Sphere((0,0,0)), pbr=True, color=color, metallic=0.0, roughness=0.2)
p.add_mesh(Sphere((0,0,1)), pbr=True, color=color, metallic=1.0, roughness=0.2)
p.view_vector((-1,0,0), (0,1,0))
p.show()
