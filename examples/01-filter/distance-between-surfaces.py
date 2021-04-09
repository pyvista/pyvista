"""
Distance Between Two Surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the average thickness between two surfaces.

For example, you might have two surfaces that represent the boundaries of
lithological layers in a subsurface geological model and you want to know the
average thickness of a unit between those boundaries.

We can compute the thickness between the two surfaces using a few different
methods. First, we will demo a method where we compute the normals of the
bottom surface, and then project a ray to the top surface to compute the
distance along the surface normals. Second, we will use a KDTree to compute
the distance from every point in the bottom mesh to it's closest point in
the top mesh.
"""
import pyvista as pv
import numpy as np

# A helper to make a random surface
def hill(seed):
    mesh = pv.ParametricRandomHills(randomseed=seed, u_res=50, v_res=50,
                                    hillamplitude=0.5)
    mesh.rotate_y(-10) # give the surfaces some tilt

    return mesh

h0 = hill(1).elevation()
h1 = hill(10)
# Shift one surface
h1.points[:,-1] += 5
h1 = h1.elevation()

###############################################################################

p = pv.Plotter()
p.add_mesh(h0, smooth_shading=True)
p.add_mesh(h1, smooth_shading=True)
p.show_grid()
p.show()

###############################################################################
# Ray Tracing Distance
# ++++++++++++++++++++
#
# Compute normals of lower surface
h0n = h0.compute_normals(point_normals=True, cell_normals=False,
                         auto_orient_normals=True)

###############################################################################
# Travel along normals to the other surface and compute the thickness on each
# vector.

h0n["distances"] = np.empty(h0.n_points)
for i in range(h0n.n_points):
    p = h0n.points[i]
    vec = h0n["Normals"][i] * h0n.length
    p0 = p - vec
    p1 = p + vec
    ip, ic = h1.ray_trace(p0, p1, first_point=True)
    dist = np.sqrt(np.sum((ip - p)**2))
    h0n["distances"][i] = dist

# Replace zeros with nans
mask = h0n["distances"] == 0
h0n["distances"][mask] = np.nan
np.nanmean(h0n["distances"])

###############################################################################
p = pv.Plotter()
p.add_mesh(h0n, scalars="distances", smooth_shading=True)
p.add_mesh(h1, color=True, opacity=0.75, smooth_shading=True)
p.show()


###############################################################################
# Nearest Neighbor Distance
# +++++++++++++++++++++++++
#
# You could also use a KDTree to compare the distance between each point of the
# upper surface and the nearest neighbor of the lower surface.
# This won't be the exact surface to surface distance, but it will be
# noticeably faster than a ray trace, especially for large surfaces.
from scipy.spatial import KDTree

tree = KDTree(h1.points)
d, idx = tree.query(h0.points )
h0["distances"] = d
np.mean(d)

###############################################################################
p = pv.Plotter()
p.add_mesh(h0, scalars="distances", smooth_shading=True)
p.add_mesh(h1, color=True, opacity=0.75, smooth_shading=True)
p.show()
