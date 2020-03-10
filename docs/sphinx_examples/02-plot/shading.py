"""
Types of Shading
~~~~~~~~~~~~~~~~

Comparison of default, flat shading vs. smooth shading.
"""
# sphinx_gallery_thumbnail_number = 2
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
