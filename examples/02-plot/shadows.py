"""
Lights and Shadows
~~~~~~~~~~~~~~~~~~

Demonstrate the usage of lights and shadows in PyVista.

"""

import pyvista
from pyvista import examples
import numpy as np

mesh = examples.download_dragon()
mesh.rotate_x(90)
mesh.rotate_z(120)

# mesh = pyvista.Sphere()
mesh = pyvista.Cube()

# mesh = examples.download_action_figure()

###############################################################################

light1 = pyvista.Light(position=(0, 0.2, 1.0),
                       focal_point=(0, 0, 0),
                       color=([1, 1.0, 0.9843, 1]),  # Color temp. 5400°K
                       intensity=0.3)

light2 = pyvista.Light(position=(0, 1.0, 1.0),
                       focal_point=(0, 0, 0),
                       color=[1, 0.83921, 0.6666, 1],  # Color temp. 2850°K
                       intensity=1)

# Add a thin box below the dragon
bounds = mesh.bounds
rnge = (bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4])
expand = 1.0
height = rnge[2] * 0.05
center = np.array(mesh.center)
center -= [0, 0, mesh.center[2] - bounds[4] + height/2]

width = rnge[0]*(1 + expand)
length = rnge[1]*(1 + expand)
# width = length = max(width, length)

base_mesh = pyvista.Cube(center,
                         width,
                         length,
                         height)

# rotate base and mesh
base_mesh.rotate_z(30)
mesh = mesh.copy()
mesh.rotate_z(30)

# create the plotter
pl = pyvista.Plotter(lighting=None)
pl.add_light(light1)
pl.add_light(light2)
pl.add_mesh(mesh, ambient=0.2, diffuse=0.5, specular=0.51, specular_power=90,
            smooth_shading=True, color='orange')
pl.add_mesh(base_mesh)
pl.background_color = 'black'
pl.enable_shadows()

# pl.camera_position = cpos
pl.camera.zoom(1.5)
print(pl.show())


###############################################################################
# shadows enabled

plotter = pyvista.Plotter(lighting=None)

for plane_y in [2, 5, 10]:
    screen = pyvista.Plane(center=(0, plane_y, 0), direction=(0, 1, 0), i_size=5,
                           j_size=5)
    plotter.add_mesh(screen, color='white')

light = pyvista.Light(position=(0, 0, 0), focal_point=(0, 1, 0),
                      color='cyan', intensity=10)
light.positional = True
light.cone_angle = 15
light.attenuation_values = (2, 0, 0)
light.show_actor()

plotter.add_light(light)
plotter.view_vector((1, -2, 2))
plotter.enable_shadows()
plotter.show()
