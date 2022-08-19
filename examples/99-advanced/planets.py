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

# https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf
# Mercury's radius is 2439.0 km
mercury = examples.planets.load_mercury(radius=2439.0, lat_resolution=150, lon_resolution=300)
# Venus's radius is 6052.0 km
venus = examples.planets.load_venus(radius=6052.0, lat_resolution=150, lon_resolution=300)
# Mars's radius is 3397.2 km
mars = examples.planets.load_mars(radius=3397.2, lat_resolution=150, lon_resolution=300)
# Jupiter's radius is 71492.0 km
jupiter = examples.planets.load_jupiter(radius=71492.0, lat_resolution=150, lon_resolution=300)
# Saturn's radius is 60268.0 km
saturn = examples.planets.load_saturn(radius=60268.0, lat_resolution=150, lon_resolution=300)
# Saturn's rings range from 7000.0 km to 80000.0 km from the surface of the planet
inner = 60268.0 + 7000.0
outer = 60268.0 + 80000.0
saturn_ring_alpha = examples.planets.load_saturn_ring_alpha(inner=inner, outer=outer, c_res=50)
# Uranus's radius is 25559.0 km
uranus = examples.planets.load_uranus(radius=25559.0, lat_resolution=150, lon_resolution=300)
# Neptune's radius is 24764.0 km
neptune = examples.planets.load_neptune(radius=24764.0, lat_resolution=150, lon_resolution=300)

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
cubemap = examples.download_cubemap_space_16k()
_ = plotter.add_actor(cubemap.to_skybox())
plotter.set_environment_texture(cubemap, True)
plotter.add_light(light)
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
plotter.add_mesh(pyvista.read(examples.planets.download_mercury_jpg(load=False)), rgb=True)
plotter.subplot(0, 1)
plotter.add_mesh(examples.planets.load_mercury())
plotter.subplot(1, 0)
plotter.add_text("Venus")
plotter.add_mesh(
    pyvista.read(examples.planets.download_venus_jpg(atmosphere=True, load=False)), rgb=True
)
plotter.subplot(1, 1)
plotter.add_mesh(examples.planets.load_venus(), texture="atmosphere")
plotter.subplot(2, 0)
plotter.add_text("Mars")
plotter.add_mesh(pyvista.read(examples.planets.download_mars_jpg(load=False)), rgb=True)
plotter.subplot(2, 1)
plotter.add_mesh(examples.planets.load_mars())
plotter.show(cpos="xy")

###############################################################################
# We can see the surface of Venus.
#

venus = examples.planets.load_venus()
plotter = pyvista.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_text("Venus Atmosphere")
plotter.add_mesh(venus, rgb=True, texture="atmosphere")
plotter.subplot(0, 1)
plotter.add_text("Venus Surface")
plotter.add_mesh(venus, rgb=True, texture="surface")
plotter.show(cpos="xy")
