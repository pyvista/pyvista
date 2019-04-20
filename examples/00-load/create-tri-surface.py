"""
Create Triangulated Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a surface from a set of points through a Delaunay triangulation.
"""

# sphinx_gallery_thumbnail_number = 2
import vtki
import numpy as np

################################################################################
#  First, create some points for the surface.

# Define a simple Gaussian surface
xx, yy = np.meshgrid(np.linspace(-200,200,20), np.linspace(-200,200,20))
A, b = 100, 100
zz = A*np.exp(-0.5*((xx/b)**2. + (yy/b)**2.))

# Get the points as a 2D NumPy array (N by 3)
points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
print(points[0:5, :])

################################################################################
# Now use those points to create a point cloud ``vtki`` data object. This will
# be encompassed in a :class:`vtki.PolyData` object.

# simply pass the numpy points to the PolyData constructor
cloud = vtki.PolyData(points)
vtki.set_plot_theme('doc')
cloud.plot(point_size=15)

################################################################################
# Now that we have a ``vtki`` data structure of the points, we can perform a
# triangulation to turn those boring discrete points into a connected surface.

surf = cloud.delaunay_2d()
surf.plot(show_edges=True)
