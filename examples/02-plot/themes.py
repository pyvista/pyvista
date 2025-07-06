"""
.. _themes_example:

Control Global and Local Plotting Themes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyVista allows you to set global and local plotting themes to easily
set default plotting parameters. This example shows how to use the
:ref:`theme_api` and :func:`~pyvista.set_plot_theme` function.

"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# %%
#  Define a simple plotting routine for comparing the themes.

mesh = examples.download_st_helens().warp_by_scalar()


def plot_example():
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.add_bounding_box()
    p.show()


# %%
# PyVista's default color theme is chosen to be generally easy on your
# eyes and is best used when working long hours on your visualization
# project.  The grey background and warm colormaps are chosen to make
# sure 3D renderings do not drastically change the brightness of your
# screen when working in dark environments.
#
# Here's an example of our default plotting theme - this is what you
# would see by default after running any of our examples locally.

pv.set_plot_theme('default')
plot_example()

# %%
# PyVista also ships with a few plotting themes:
#
# * ``'ParaView'``: this is designed to mimic ParaView's default plotting theme.
# * ``'dark'``: this is designed to be night-mode friendly with dark backgrounds and color schemes.
# * ``'document'``: this is built for use in document style plotting and making publication
#   quality figures.

# %%
# Demo the ``'ParaView'`` theme.

pv.set_plot_theme('paraview')

plot_example()


# %%
# Demo the ``'dark'`` theme.

pv.set_plot_theme('dark')

plot_example()

# %%
# Demo the ``'document'`` theme.  This theme is used on our online examples.

pv.set_plot_theme('document')

plot_example()

# %%
# Note that you can also use color gradients for the background of the plotting
# window.
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show_grid()
# Here we set the gradient
plotter.set_background('royalblue', top='aliceblue')
cpos = plotter.show()


# %%
# Modifying the Global Theme
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can control how meshes are displayed by setting individual
# parameters when plotting like ``mesh.plot(show_edges=True)``, or by
# setting a global theme.  You can also control individual parameters
# how all meshes are displayed by default via ``pyvista.global_theme``.
#
# Here, we print out the current global defaults for all ``pyvista``
# meshes.  These values have been changed by the previous "Document"
# theme.

pv.global_theme


# %%
# By default, edges are not shown on meshes unless explicitly
# specified when plotting a mesh via ``show_edges=True``.  You can
# change this default behavior globally by changing the default
# parameter.

pv.global_theme.show_edges = True
cpos = pv.Sphere().plot()


# %%
# You can reset pyvista to default behavior with ``restore_defaults``.
# Note that the figure's color was reset to the default "white" color
# rather than the 'lightblue' color default with the document theme.  Under
# the hood, each theme applied changes the global plot defaults stored
# within ``pyvista.global_theme.``

pv.global_theme.restore_defaults()
cpos = pv.Sphere().plot()


# %%
# Creating a Custom Theme and Applying it Globally
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can create a custom theme by modifying one of the existing
# themes and then loading it into the global plotting defaults.
#
# Here, we create a dark theme that plots meshes red by default while
# showing edges.

from pyvista import themes

my_theme = themes.DarkTheme()
my_theme.color = 'red'
my_theme.lighting = False
my_theme.show_edges = True
my_theme.axes.box = True

pv.global_theme.load_theme(my_theme)
cpos = pv.Sphere().plot()


# %%
# Creating a Custom Theme and Applying it to a Single Plotter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In this example, we create a custom theme from the base "default"
# theme and then apply it to a single plotter.  Note that this does
# not change the behavior of the global "defaults", which are still
# set to the modified ``DarkTheme``.
#
# This approach carries the advantage that you can maintain several
# themes and apply them to one or more plotters.

from pyvista import themes

my_theme = themes.DocumentTheme()
my_theme.color = 'black'
my_theme.lighting = True
my_theme.show_edges = True
my_theme.edge_color = 'white'
my_theme.background = 'white'

cpos = pv.Sphere().plot(theme=my_theme)


# %%
# Alternatively, set the theme of an instance of ``Plotter``.

pl = pv.Plotter(theme=my_theme)
# pl.theme = my_theme  # alternatively use the setter
pl.add_mesh(pv.Cube())
cpos = pl.show()


# %%
# Reset to use the document theme
pv.set_plot_theme('document')
# %%
# .. tags:: plot
