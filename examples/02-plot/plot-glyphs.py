"""
Plotting Glyphs
~~~~~~~~~~~~~~~

Use vectors in a dataset to plot and orient glyphs/geometric objects.
"""

# sphinx_gallery_thumbnail_number = 3
import vtki
import numpy as np

################################################################################
# Glyphying can be done via the :func:`vtki.DataSetFilters.glyph` filter

# Make a grid:
x, y, z = np.meshgrid(np.linspace(-5, 5, 20),
                  np.linspace(-5, 5, 20),
                  np.linspace(-5, 5, 5))

grid = vtki.StructuredGrid(x, y, z)

vectors = np.sin(grid.points)**3


# Compute a direction for the vector field
grid.point_arrays['mag'] = np.linalg.norm(vectors, axis=1)
grid.point_arrays['vec'] = vectors

# plot using the plotting class
p = vtki.Plotter()
p.add_mesh(grid.glyph(orient='vec', scale='mag', factor=1))
p.show()

################################################################################
# Another approach is to load the vectors directly to the grid object and then
# access the :attr:`vtki.Common.arrows` property.

sphere = vtki.Sphere(radius=3.14)

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
p = vtki.Plotter()
p.add_mesh(sphere.arrows, lighting=False, stitle='Vector Magnitude')
p.add_mesh(sphere, color='grey', ambient=0.6, opacity=0.5, show_edges=False)
p.show()
