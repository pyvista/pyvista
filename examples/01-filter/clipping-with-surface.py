"""
Clipping with a Surface
~~~~~~~~~~~~~~~~~~~~~~~

Clip any PyVista dataset by a :class:`pyvista.PolyData` surface mesh using
the :func:`pyvista.DataSet.Filters.clip_surface` filter.
"""
# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples
import numpy as np


###############################################################################
surface = pv.Cone(direction=(0,0,-1), height=3.0, radius=1,
                  resolution=50, capping=False)

# Make a gridded dataset
n = 51
xx = yy = zz = 1 - np.linspace(0, n, n) * 2 / (n-1)
dataset = pv.RectilinearGrid(xx, yy, zz)

# Preview the problem
p = pv.Plotter()
p.add_mesh(surface, color='w', label='Surface')
p.add_mesh(dataset, color='gold', show_edges=True,
           opacity=0.75, label='To Clip')
p.add_legend()
p.show()

###############################################################################
# Clip the rectilinear grid dataset using the :class:`pyvista.PolyData`
# surface mesh:
clipped = dataset.clip_surface(surface, invert=False)

# Visualize the results
p = pv.Plotter()
p.add_mesh(surface, color='w', opacity=0.75, label='Surface')
p.add_mesh(clipped, color='gold', show_edges=True, label="clipped")
p.add_legend()
p.show()


###############################################################################
# Here is another example of clipping a mesh by a surface. This time, we'll
# generate a :class:`pyvista.UniformGrid` around a topography surface and then
# clip that grid using the surface to create a closed 3D model of the surface
surface = examples.load_random_hills()

# Create a grid around that surface
grid = pv.create_grid(surface)

# Clip the grid using the surface
model = grid.clip_surface(surface)

# Compute height and display it
model.elevation().plot()
