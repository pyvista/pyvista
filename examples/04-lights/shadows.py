"""
.. _light_shadows_example:

Shadows
~~~~~~~

Demonstrate the usage of lights and shadows in PyVista.
"""

import numpy as np

# sphinx_gallery_thumbnail_number = 2
import pyvista
from pyvista import examples

mesh = examples.download_dragon()
mesh.rotate_x(90, inplace=True)
mesh.rotate_z(120, inplace=True)


###############################################################################
# Using two lights, plot the Stanford Dragon with shadows.
#
# .. Note::
#    VTK has known issues when rendering shadows on certain window
#    sizes.  Be prepared to experiment with the ``window_size``
#    parameter.  An initial window size of ``(1000, 1000)`` seems to
#    work well, which can be manually resized without issue.

light1 = pyvista.Light(
    position=(0, 0.2, 1.0),
    focal_point=(0, 0, 0),
    color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
    intensity=0.3,
)

light2 = pyvista.Light(
    position=(0, 1.0, 1.0),
    focal_point=(0, 0, 0),
    color=[1.0, 0.83921, 0.6666, 1.0],  # Color temp. 2850 K
    intensity=1,
)

# Add a thin box below the mesh
bounds = mesh.bounds
rnge = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

expand = 1.0
height = rnge[2] * 0.05
center = np.array(mesh.center)
center -= [0, 0, mesh.center[2] - bounds[4] + height / 2]

width = rnge[0] * (1 + expand)
length = rnge[1] * (1 + expand)
base_mesh = pyvista.Cube(center, width, length, height)

# rotate base and mesh to get a better view
base_mesh.rotate_z(30, inplace=True)
mesh.rotate_z(30, inplace=True)

# create the plotter with custom lighting
pl = pyvista.Plotter(lighting=None, window_size=(800, 800))
pl.add_light(light1)
pl.add_light(light2)
pl.add_mesh(
    mesh,
    ambient=0.2,
    diffuse=0.5,
    specular=0.5,
    specular_power=90,
    smooth_shading=True,
    color='orange',
)
pl.add_mesh(base_mesh)
pl.enable_shadows()
pl.camera.zoom(1.5)
pl.show()


###############################################################################
# Show light penetrating several planes.  Adjust the light intensity
# and the ``shadow_attenuation`` to change how many planes the
# light can go through.

plotter = pyvista.Plotter(lighting=None, window_size=(800, 800))

# add several planes
for plane_y in [2, 5, 10]:
    screen = pyvista.Plane(center=(0, plane_y, 0), direction=(0, 1, 0), i_size=5, j_size=5)
    plotter.add_mesh(screen, color='white')

light = pyvista.Light(
    position=(0, 0, 0),
    focal_point=(0, 1, 0),
    color='cyan',
    intensity=15,
    positional=True,
    cone_angle=15,
    attenuation_values=(2, 0, 0),
)
light.show_actor()

plotter.add_light(light)
plotter.view_vector((1, -2, 2))
plotter.enable_shadows()
plotter.show()


###############################################################################
# Here, we use a lower shadow_attenuation value to demonstrate how the
# light can travel through more planes.

plotter = pyvista.Plotter(lighting=None, window_size=(800, 800))

# add several planes
for plane_y in [2, 5, 10]:
    screen = pyvista.Plane(center=(0, plane_y, 0), direction=(0, 1, 0), i_size=5, j_size=5)
    plotter.add_mesh(screen, color='white')

light = pyvista.Light(
    position=(0, 0, 0),
    focal_point=(0, 1, 0),
    color='cyan',
    intensity=15,
    cone_angle=15,
    shadow_attenuation=0.95,
    positional=True,
    attenuation_values=(2, 0, 0),
)
light.show_actor()

plotter.add_light(light)
plotter.view_vector((1, -2, 2))
plotter.enable_shadows()
plotter.show()
