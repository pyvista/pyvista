"""
Create Triangulated Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a surface from a set of points through a Delaunay triangulation.
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import numpy as np
from pyvista import examples

###############################################################################
# Example A
# +++++++++
#
# First, create some points for the surface.

# Define a simple Gaussian surface
n = 20
x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
xx, yy = np.meshgrid(x, y)
A, b = 100, 100
zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

# Get the points as a 2D NumPy array (N by 3)
points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
points[0:5, :]

###############################################################################
# Now use those points to create a point cloud PyVista data object. This will
# be encompassed in a :class:`pyvista.PolyData` object.

# simply pass the numpy points to the PolyData constructor
cloud = pv.PolyData(points)
cloud.plot(point_size=15)

###############################################################################
# Now that we have a PyVista data structure of the points, we can perform a
# triangulation to turn those boring discrete points into a connected surface.

surf = cloud.delaunay_2d()
surf.plot(show_edges=True)


###############################################################################
# Example B
# +++++++++
#

cpos = [(1428.9156647715076, -809.393306371347, -986.0854935143382),
 (131.81156409644166, 11.062412559200027, -7.405967645460279),
 (0.6932886107381906, 0.3348516198391111, 0.6381420648349653)]

original_surface = examples.download_delaunay_example()
cloud = pv.PolyData(original_surface.points)
cloud.plot(point_size=15, cpos=cpos, show_grid=True)

###############################################################################
surf = cloud.delaunay_2d()
surf.plot(cpos=cpos, show_grid=True, )
