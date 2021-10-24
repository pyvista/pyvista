"""
.. _marching_cubes_example:

Marching Cubes
~~~~~~~~~~~~~~

Generate a surface from a scalar field using the flying edges of
marching cubes algorithm.

Special thanks to ... for developing the scalar functions.

"""
import pyvista as pv
import numpy as np


###############################################################################
# Spider Cage
# ~~~~~~~~~~~
# Use the marching cubes algorithm to extract the isosurface
# generated from the spider cage function.

a = 0.9
def spider_cage(x, y, z):
    x2 = x * x
    y2 = y * y
    x2_y2 = x2 + y2
    return (
        np.sqrt((x2 - y2)**2 / x2_y2 + 3 * (z * np.sin(a))**2) - 3)**2 + 6 * (
        np.sqrt((x * y)**2 / x2_y2 + (z * np.cos(a))**2) - 1.5
    )**2


n = 100
x_min, y_min, z_min = -5, -5, -3
grid = pv.UniformGrid(
    (n, n, n),
    (abs(x_min)/n*2, abs(y_min)/n*2, abs(z_min)/n*2),
    (x_min, y_min, z_min),
)
x, y, z = grid.points.T
values = spider_cage(x, y, z)
mesh = grid.contour(1, values, method='marching_cubes', rng=[1, 0])
dist = np.linalg.norm(mesh.points, axis=1)
mesh.plot(
    scalars=dist, smooth_shading=True, specular=5,
    cmap="plasma", show_scalar_bar=False
)


