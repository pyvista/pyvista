"""
.. _glyph_example:

Plotting Glyphs
~~~~~~~~~~~~~~~

Use vectors in a dataset to plot and orient glyphs/geometric objects.
"""

# sphinx_gallery_thumbnail_number = 1
import pyvista as pv
from pyvista import examples
import numpy as np

################################################################################
# Glyphying can be done via the :func:`pyvista.DataSetFilters.glyph` filter

mesh = examples.download_carotid().threshold(145, scalars='scalars')

# Make a geometric obhect to use as the glyph
geom = pv.Arrow() # This could be any dataset

# Perform the glyph
glyphs = mesh.glyph(orient='vectors', scale='scalars', factor=0.005, geom=geom)

# plot using the plotting class
p = pv.Plotter()
p.add_mesh(glyphs)
# Set a cool camera position
p.camera_position = [(84.58052237950857, 77.76332116787425, 27.208569926456548),
 (131.39486171068918, 99.871379394528, 20.082859824932008),
 (0.13483731007732908, 0.033663777790747404, 0.9902957385932576)]
p.show()

################################################################################
# Another approach is to load the vectors directly to the mesh object and then
# access the :attr:`pyvista.Common.arrows` property.

sphere = pv.Sphere(radius=3.14)

# make cool swirly pattern
vectors = np.vstack((np.sin(sphere.points[:, 0]),
np.cos(sphere.points[:, 1]),
np.cos(sphere.points[:, 2]))).T

# add and scale
sphere.vectors = vectors*0.3

# plot just the arrows
sphere.arrows.plot()

################################################################################

# plot the arrows and the sphere
p = pv.Plotter()
p.add_mesh(sphere.arrows, lighting=False, stitle='Vector Magnitude')
p.add_mesh(sphere, color='grey', ambient=0.6, opacity=0.5, show_edges=False)
p.show()
