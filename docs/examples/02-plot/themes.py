"""
Change the Theme
~~~~~~~~~~~~~~~~

PyVista has a few coloring themes for you to choose!
"""
import pyvista as pv
from pyvista import examples

###############################################################################
#  Define a simple plotting routine for comparing the themes

mesh = examples.download_st_helens().warp_by_scalar()


def plot_example():
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.add_bounding_box()
    return p.show()


###############################################################################
# PyVista's default color theme is chosen to be generally easy on your eyes
# and is best used when working long hours on your visualization project.
# The grey background and warm colormaps are chosen to make sure 3D renderings
# do not drastically change the brightness of your screen when working in dark
# environments.
#
# Here's an example of our default plotting theme - this is what you would see
# by default after running any of our examnples.

pv.set_plot_theme("default")

plot_example()

###############################################################################
# PyVista also ships with a few plotting themes:
#
# * ``'ParaView'``: this is designed to mimic ParaView's default plotting theme
# * ``'night'``: this is designed to be night-mode friendly with dark backgrounds and color schemes
# * ``'document'``: this is built for use in document style plotting and making publication quality figures

###############################################################################
# Demo the ``'ParaView'`` theme

pv.set_plot_theme("ParaView")

plot_example()


###############################################################################
# Demo the ``'night'`` theme

pv.set_plot_theme("night")

plot_example()

###############################################################################
# Demo the ``'document'`` theme


pv.set_plot_theme("document")

plot_example()

###############################################################################
# Note that you can also use color gradients for the background of the plotting
# window!
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show_grid()
# Here we set the gradient
plotter.set_background("royalblue", top="aliceblue")
plotter.show()
