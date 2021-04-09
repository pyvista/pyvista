"""
.. _glyph_example:

Plotting Glyphs (Vectors or PolyData)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use vectors in a dataset to plot and orient glyphs/geometric objects.
"""

# sphinx_gallery_thumbnail_number = 4
import pyvista as pv
from pyvista import examples
import numpy as np

###############################################################################
# Glyphying can be done via the :func:`pyvista.DataSetFilters.glyph` filter

mesh = examples.download_carotid().threshold(145, scalars="scalars")

# Make a geometric object to use as the glyph
geom = pv.Arrow()  # This could be any dataset

# Perform the glyph
glyphs = mesh.glyph(orient="vectors", scale="scalars", factor=0.005, geom=geom)

# plot using the plotting class
p = pv.Plotter()
p.add_mesh(glyphs)
# Set a cool camera position
p.camera_position = [
    (84.58052237950857, 77.76332116787425, 27.208569926456548),
    (131.39486171068918, 99.871379394528, 20.082859824932008),
    (0.13483731007732908, 0.033663777790747404, 0.9902957385932576),
]
p.show()

###############################################################################
# Another approach is to load the vectors directly to the mesh object and then
# access the :attr:`pyvista.DataSet.arrows` property.

sphere = pv.Sphere(radius=3.14)

# make cool swirly pattern
vectors = np.vstack(
    (
        np.sin(sphere.points[:, 0]),
        np.cos(sphere.points[:, 1]),
        np.cos(sphere.points[:, 2]),
    )
).T

# add and scale
sphere.vectors = vectors * 0.3

# plot just the arrows
sphere.arrows.plot(scalars='GlyphScale')

###############################################################################

# plot the arrows and the sphere
p = pv.Plotter()
p.add_mesh(sphere.arrows, scalars='GlyphScale', lighting=False,
           scalar_bar_args={'title': "Vector Magnitude"})
p.add_mesh(sphere, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
p.show()


###############################################################################
# Subset of Glyphs
# ++++++++++++++++
#
# Sometimes you might not want glyphs for every node in the input dataset. In
# this case, you can choose to build glyphs for a subset of the input dataset
# by using a merging tolerance. Here we specify a merging tolerance of five
# percent which equates to five percent of the bounding box's length.

# Example dataset with normals
mesh = examples.load_random_hills()

# create a subset of arrows using the glyph filter
arrows = mesh.glyph(scale="Normals", orient="Normals", tolerance=0.05)

p = pv.Plotter()
p.add_mesh(arrows, color="black")
p.add_mesh(mesh, scalars="Elevation", cmap="terrain")
p.show()
