"""
.. _planets_example:

3D Earth and Celestial Bodies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates high-resolution renderings of the Earth and the major celestial bodies
in our solar system for astrodynamics applications.
This example is inspired by `planet3D-MATLAB <https://github.com/tamaskis/planet3D-MATLAB>`_ .


Note that this example author doesn't know anything about astronomy. There may
be inaccuracies in the representation. The purpose of this example is for
inspiration only.

"""
import pyvista
from pyvista import examples

###############################################################################
# Celestial bodies with the Milky Way in the background.


# Light of the sun in the +x axis direction.

light = pyvista.Light()
light.set_direction_angle(30, -20)

# Load planet

mercury = examples.load_mercury()
venus = examples.load_venus()
mars = examples.load_mars()
jupiter = examples.load_jupiter()
saturn = examples.load_saturn()
saturn_ring_alpha = examples.load_saturn_ring_alpha()
uranus = examples.load_uranus()
neptune = examples.load_neptune()

# Move planet position (Numbers have no meaning. The planets are laid out for
# easy viewing.).

mercury.translate((0.0, 0.0, 0.0), inplace=True)
venus.translate((-15000.0, 0.0, 0.0), inplace=True)
mars.translate((-45000.0, 0.0, 0.0), inplace=True)
jupiter.translate((-150000.0, 0.0, 0.0), inplace=True)
saturn.translate((-400000.0, 0.0, 0.0), inplace=True)
saturn_ring_alpha.translate((-400000.0, 0.0, 0.0), inplace=True)
uranus.translate((-600000.0, 0.0, 0.0), inplace=True)
neptune.translate((-700000.0, 0.0, 0.0), inplace=True)

# Add planets to Plotter.

plotter = pyvista.Plotter(lighting="none")
plotter.add_light(light)
plotter.add_background_image(examples.download_milkyway_jpg())
plotter.add_mesh(mercury)
plotter.add_mesh(venus)
plotter.add_mesh(mars)
plotter.add_mesh(jupiter)
plotter.add_mesh(saturn)
plotter.add_mesh(saturn_ring_alpha)
plotter.add_mesh(uranus)
plotter.add_mesh(neptune)
plotter.camera.zoom(1.5)
plotter.show()

###############################################################################
# `Textures in this pack are based on NASA <https://www.solarsystemscope.com/textures/https://www.solarsystemscope.com/textures/>`_
# elevation and imagery data. Colors and shades of the textures are tuned
# according to true-color photos made by Messenger, Viking and Cassini
# spacecrafts, and, of course, the Hubble Space Telescope.

plotter = pyvista.Plotter(shape=(3, 2))
plotter.subplot(0, 0)
plotter.add_text("Mercury")
plotter.add_mesh(pyvista.read(examples.download_mercury_jpg()), rgb=True)
plotter.subplot(0, 1)
plotter.add_mesh(examples.load_mercury())
plotter.subplot(1, 0)
plotter.add_text("Venus")
plotter.add_mesh(pyvista.read(examples.download_venus_jpg(atmosphere=True)), rgb=True)
plotter.subplot(1, 1)
plotter.add_mesh(examples.load_venus(), texture="atmosphere")
plotter.subplot(2, 0)
plotter.add_text("Mars")
plotter.add_mesh(pyvista.read(examples.download_mars_jpg()), rgb=True)
plotter.subplot(2, 1)
plotter.add_mesh(examples.load_mars())
plotter.show(cpos="xy")

###############################################################################
# We can see the surface of Venus.
#

venus = examples.load_venus()
plotter = pyvista.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_text("Venus Atmosphere")
plotter.add_mesh(venus, rgb=True, texture="atmosphere")
plotter.subplot(0, 1)
plotter.add_text("Venus Surface")
plotter.add_mesh(venus, rgb=True, texture="surface")
plotter.show(cpos="xy")
