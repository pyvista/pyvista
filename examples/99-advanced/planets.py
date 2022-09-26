"""
.. _planets_example:

3D Earth and Celestial Bodies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example is inspired by `planet3D-MATLAB <https://github.com/tamaskis/planet3D-MATLAB>`_.


.. note::
   The purpose of this example is to demonstrate plotting celestial bodies and
   may lack astronomical precision. There maybe inaccuracies in the
   representation, so please take care when reusing or repurposing this example.

   Please take a look at libraries like `astropy <https://www.astropy.org/>`_
   if you wish to use Python for astronomical calculations.

"""
import pyvista
from pyvista import examples

###############################################################################
# Plot the Solar System with Stars in the Background
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This section relies on calculations in `Visualizing Celestial Bodies in 3D
# <https://tamaskis.github.io/files/Visualizing_Celestial_Bodies_in_3D.pdf>`_.


# Light of the Sun.
light = pyvista.Light()
light.set_direction_angle(30, -20)

# Load planets

kwargs = dict(lat_resolution=150, lon_resolution=300)
mercury = examples.planets.load_mercury(radius=2439.0, **kwargs)
venus = examples.planets.load_venus(radius=6052.0, **kwargs)
mars = examples.planets.load_mars(radius=3397.2, **kwargs)
jupiter = examples.planets.load_jupiter(radius=71492.0, **kwargs)
saturn = examples.planets.load_saturn(radius=60268.0, **kwargs)
# Saturn's rings range from 7000.0 km to 80000.0 km from the surface of the planet
inner = 60268.0 + 7000.0
outer = 60268.0 + 80000.0
saturn_ring_alpha = examples.planets.load_saturn_ring_alpha(inner=inner, outer=outer, c_res=50)
uranus = examples.planets.load_uranus(radius=25559.0, **kwargs)
neptune = examples.planets.load_neptune(radius=24764.0, **kwargs)
pluto = examples.planets.load_pluto(radius=1151.0, **kwargs)

# Move planets to a nice position for the plotter. These numbers are not grounded in
# reality and are for demonstration purposes only.
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
plotter.add_mesh(mercury, smooth_shading=True)
plotter.add_mesh(venus, smooth_shading=True)
plotter.add_mesh(mars, smooth_shading=True)
plotter.add_mesh(jupiter, smooth_shading=True)
plotter.add_mesh(saturn, smooth_shading=True)
plotter.add_mesh(saturn_ring_alpha, smooth_shading=True)
plotter.add_mesh(uranus, smooth_shading=True)
plotter.add_mesh(neptune, smooth_shading=True)
plotter.add_mesh(pluto, smooth_shading=True)
plotter.camera.zoom(1.5)
plotter.show()

###############################################################################
# Plot the Planets in Separate Subplots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each planet here is in a different subplot. The planet's textures are from
# `Solar Textures <https://www.solarsystemscope.com/textures/>`_.

plotter = pyvista.Plotter(shape=(3, 2))
plotter.subplot(0, 0)
plotter.add_text("Mercury")
plotter.add_mesh(examples.planets.download_mercury_texture(load=True), rgb=True)
plotter.subplot(0, 1)
plotter.add_mesh(examples.planets.download_mercury_texture(load=True), rgb=True)
plotter.subplot(1, 0)
plotter.add_text("Venus")
plotter.add_mesh(examples.planets.download_venus_texture(atmosphere=True, load=True), rgb=True)
plotter.subplot(1, 1)
plotter.add_mesh(examples.planets.download_venus_texture(atmosphere=True, load=True), rgb=True)
plotter.subplot(2, 0)
plotter.add_text("Mars")
plotter.add_mesh(examples.planets.download_mars_texture(load=True), rgb=True)
plotter.subplot(2, 1)
plotter.add_mesh(examples.planets.load_mars())
plotter.show(cpos="xy")

###############################################################################
# Plot the Surface of Venus
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we plot Venus with and without its atmosphere.

venus = examples.planets.load_venus()
plotter = pyvista.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_text("Venus Atmosphere")
plotter.add_mesh(venus, texture="atmosphere", smooth_shading=True)
plotter.subplot(0, 1)
plotter.add_text("Venus Surface")
plotter.add_mesh(venus, texture="surface", smooth_shading=True)
plotter.show(cpos="xy")
