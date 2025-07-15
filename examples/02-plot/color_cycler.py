"""
.. _color_cycler_example:

Color Cycling
~~~~~~~~~~~~~

Cycle through colors when sequentially adding meshes to a plotter.
"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

# %%
# Many plotting libraries like Matplotlib cycle through a predefined list of
# colors to colorize the data being added to the graphic. PyVista supports
# this in much the same way as Matplotlib.

# sphinx_gallery_thumbnail_number = 1
from __future__ import annotations

import pyvista as pv

# %%
# Turn on color cycling in PyVista's theme and set it to use the default
# cycler.
pv.global_theme.color_cycler = 'default'

# %%
# List the available colors in the cycler
pv.global_theme.color_cycler

# %%
# Create a plotter and add data to the scene. You'll notice that each
# ``add_mesh`` call iterates over the colors in ``pv.global_theme.color_cycler``
pl = pv.Plotter()
pl.add_mesh(pv.Cone(center=(0, 0, 0)))
pl.add_mesh(pv.Cube(center=(1, 0, 0)))
pl.add_mesh(pv.Sphere(center=(1, 1, 0)))
pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))
pl.show()

# %%
# Reset the theme to not use a cycler and instead set on individual plotters.
pv.global_theme.color_cycler = None

# %%
# If you do not want to set a global color cycler but instead just want to
# use a cycler for a single plotter, you can set this on with
# :func:`set_color_cycler() <pyvista.Plotter.set_color_cycler>`.
pl = pv.Plotter()

# Set to iterate over Red, Green, and Blue
pl.set_color_cycler(['red', 'green', 'blue'])

pl.add_mesh(pv.Cone(center=(0, 0, 0)))  # red
pl.add_mesh(pv.Cube(center=(1, 0, 0)))  # green
pl.add_mesh(pv.Sphere(center=(1, 1, 0)))  # blue
pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))  # red again
pl.show()

# %%
# Further, you can control this on a per-renderer basis by calling
# :func:`set_color_cycler() <pyvista.Renderer.set_color_cycler>` on the active
# ``renderer``.
pl = pv.Plotter(shape=(1, 2))

pl.subplot(0, 0)
pl.renderer.set_color_cycler('default')
pl.add_mesh(pv.Cone(center=(0, 0, 0)))
pl.add_mesh(pv.Cube(center=(1, 0, 0)))
pl.add_mesh(pv.Sphere(center=(1, 1, 0)))
pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))

pl.subplot(0, 1)
pl.renderer.set_color_cycler(['magenta', 'seagreen', 'aqua', 'orange'])
pl.add_mesh(pv.Cone(center=(0, 0, 0)))
pl.add_mesh(pv.Cube(center=(1, 0, 0)))
pl.add_mesh(pv.Sphere(center=(1, 1, 0)))
pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))

pl.link_views()
pl.view_isometric()
pl.show()


# %%
# You can also change the colors of actors after they are added to the scene.
#
# ProTip: you could place the for-loop below in an event callback for a key
# event to cycle through the colors on-demand. Or better yet, have your cycler
# randomly select colors.
from cycler import cycler

pl = pv.Plotter()
pl.add_mesh(pv.Cone(center=(0, 0, 0)))
pl.add_mesh(pv.Cube(center=(1, 0, 0)))
pl.add_mesh(pv.Sphere(center=(1, 1, 0)))
pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))

colors = cycler('color', ['lightcoral', 'seagreen', 'aqua', 'firebrick'])()

for actor in pl.renderer.actors.values():
    if isinstance(actor, pv.Actor):
        actor.prop.color = next(colors)['color']

pl.show()
# %%
# .. tags:: plot
