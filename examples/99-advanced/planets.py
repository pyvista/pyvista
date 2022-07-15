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

###############################################################################
# Celestial bodies with the Milky Way in the background.

from pyvista import examples

light = pyvista.Light()
light.set_direction_angle(30, -20)

plotter = pyvista.Plotter(lighting="none")
plotter.add_light(light)
plotter.add_background_image(examples.download_milkyway_jpg())
plotter.add_mesh(examples.load_mercury())
plotter.add_mesh(examples.load_venus().translate((-15000.0, 0.0, 0.0), inplace=True))
plotter.add_mesh(examples.load_mars().translate((-45000.0, 0.0, 0.0), inplace=True))
plotter.add_mesh(examples.load_jupiter().translate((-150000.0, 0.0, 0.0), inplace=True))
plotter.add_mesh(examples.load_saturn().translate((-400000.0, 0.0, 0.0), inplace=True))
plotter.add_mesh(examples.load_saturn_ring_alpha().translate((-400000.0, 0.0, 0.0), inplace=True))
plotter.add_mesh(examples.load_uranus().translate((-600000.0, 0.0, 0.0), inplace=True))
plotter.add_mesh(examples.load_neptune().translate((-700000.0, 0.0, 0.0), inplace=True))
plotter.camera.zoom(1.5)
plotter.show()

###############################################################################
# `Textures in this pack are based on NASA <https://www.solarsystemscope.com/textures/https://www.solarsystemscope.com/textures/>`_
# elevation and imagery data. Colors and shades of the textures are tuned
# accordng to true-color photos made by Messenger, Viking and Cassini
# spacecrafts, and, of course, the Hubble Space Telescope.

plotter = pyvista.Plotter(shape=(3, 2))
plotter.subplot(0, 0)
plotter.add_text("Mercury")
plotter.add_mesh(pyvista.read(examples.download_mercury_jpg()), rgb=True)
plotter.subplot(0, 1)
plotter.add_mesh(examples.load_mercury())
plotter.subplot(1, 0)
plotter.add_text("Venus")
plotter.add_mesh(pyvista.read(examples.download_venus_jpg()), rgb=True)
plotter.subplot(1, 1)
plotter.add_mesh(examples.load_venus())
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
