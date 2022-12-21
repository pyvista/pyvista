"""
.. _color_cycler_example:

Color Cycling
~~~~~~~~~~~~~

Cycle through colors when sequentially adding meshes to a plotter.
"""
###############################################################################
# Many plotting libraries like Matplotlib cycle through a predefined list of
# colors to colorize the data being added to the graphic. PyVista supports
# this in much the same way as Matplotlib.
#
# .. note::
#    This requires matplotlib (or at least cycler) to be installed.

# sphinx_gallery_thumbnail_number = 1
import pyvista as pv

###############################################################################
# Turn on color cycling in PyVista's theme and set it to use the default
# cycler.
pv.global_theme.color_cycler = 'default'

###############################################################################
# List the available colors in the cycler
pv.global_theme.color_cycler

###############################################################################
# Create a plotter and add data to the scene. You'll notice that each
# ``add_mesh`` call iterates over the colors in ``pv.global_theme.color_cycler``
p = pv.Plotter()
p.add_mesh(pv.Cone(center=(0, 0, 0)))
p.add_mesh(pv.Cube(center=(1, 0, 0)))
p.add_mesh(pv.Sphere(center=(1, 1, 0)))
p.add_mesh(pv.Cylinder(center=(0, 1, 0)))
p.show()

###############################################################################
# Reset the theme to not use a cycler and instead set on individual plotters.
pv.global_theme.color_cycler = None

###############################################################################
# If you do not want to set a global color cycler but instead just want to
# use a cycler for a single plotter, you can set this on with
# ``set_color_cycler``.
p = pv.Plotter()

# Set to iterate over Red, Green, and Blue
p.set_color_cycler(['red', 'green', 'blue'])

p.add_mesh(pv.Cone(center=(0, 0, 0)))  # red
p.add_mesh(pv.Cube(center=(1, 0, 0)))  # green
p.add_mesh(pv.Sphere(center=(1, 1, 0)))  # blue
p.add_mesh(pv.Cylinder(center=(0, 1, 0)))  # red again
p.show()

###############################################################################
# Further, you can control this on a per-renderer-basis by calling ``set_color_cycler()`` on the activer ``renderer``.
p = pv.Plotter(shape=(1, 2))

p.subplot(0, 0)
p.renderer.set_color_cycler('default')
p.add_mesh(pv.Cone(center=(0, 0, 0)))
p.add_mesh(pv.Cube(center=(1, 0, 0)))
p.add_mesh(pv.Sphere(center=(1, 1, 0)))
p.add_mesh(pv.Cylinder(center=(0, 1, 0)))

p.subplot(0, 1)
p.renderer.set_color_cycler(['magenta', 'seagreen', 'aqua', 'orange'])
p.add_mesh(pv.Cone(center=(0, 0, 0)))
p.add_mesh(pv.Cube(center=(1, 0, 0)))
p.add_mesh(pv.Sphere(center=(1, 1, 0)))
p.add_mesh(pv.Cylinder(center=(0, 1, 0)))

p.link_views()
p.view_isometric()
p.show()
