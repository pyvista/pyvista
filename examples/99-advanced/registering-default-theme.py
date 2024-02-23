"""
.. _default_themes:

Registering Default Themes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Downstream packages may want to register custom themes for users to choose from directly in PyVista.
PyVista allows developers to register themes to show as available as a custom theme within PyVista.

"""

###############################################################################
# The commonly available default themes can be inspected.
import pyvista as pv
from pyvista.plotting.themes import Theme

Theme.defaults()

###############################################################################
# Make a plot for comparison later.

pv.set_plot_theme("vtk")
pv.plot(pv.Sphere())


###############################################################################
# Registering a custom themes is done through specifying a dictionary.

edge_theme_dict = {"show_edges": True}

###############################################################################
# Register theme as a default with name ``edge_theme`` and a docstring.

Theme.register_default(
    "edge_theme",
    edge_theme_dict,
    doc="""Theme that shows edges.""",
)

###############################################################################
# Activate new default theme and replot.
pv.set_plot_theme("edge_theme")
pv.plot(pv.Sphere())

###############################################################################
# Users can also get the theme for further customization.
custom_edge_theme = Theme.edge_theme()
custom_edge_theme.color = 'blue'
pv.global_theme.load_theme(custom_edge_theme)

###############################################################################
# Reset to use the document theme.
pv.set_plot_theme("document")
