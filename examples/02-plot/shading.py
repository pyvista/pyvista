"""
Types of Shading
~~~~~~~~~~~~~~~~

Use a custom built colormap when plotting scalar values.
"""

import pyvista

pyvista.set_plot_theme("document")
###############################################################################
# PyVista supports two types of shading, flat and smooth shading that uses
# VTK's Phong shading algorithm.
#
# This is a plot with the default flat shading:
sphere = pyvista.Sphere()
sphere.plot(color="w")

###############################################################################
# Here's the same sphere with smooth shading:
sphere.plot(color="w", smooth_shading=True)
