"""
Resampling & Interpolating
~~~~~~~~~~~~~~~~~~~~~~~~~~

Resample one mesh's point/cell arrays onto another meshes nodes.
"""
###############################################################################
# This example will resample a volumetric mesh's  scalar data onto the surface
# of a sphere contained in that volume.

# sphinx_gallery_thumbnail_number = 4
import pyvista as pv
from pyvista import examples

###############################################################################
# Query a grids points onto a sphere
mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
data_to_probe = examples.load_uniform()

###############################################################################
# Plot the two datasets
p = pv.Plotter()
p.add_mesh(mesh, color=True)
p.add_mesh(data_to_probe, opacity=0.5)
p.show()

###############################################################################
# Run the algorithm and plot the result
result = mesh.sample(data_to_probe)

# Plot result
name = "Spatial Point Data"
result.plot(scalars=name, clim=data_to_probe.get_data_range(name))


###############################################################################
# Interpolate
# +++++++++++
#
# Resample the points' arrays onto a surface using an interpolation from a Gaussian Kernel

# Download sample data
surface = examples.download_saddle_surface()
points = examples.download_sparse_points()


p = pv.Plotter()
p.add_mesh(points, point_size=30.0, render_points_as_spheres=True)
p.add_mesh(surface)
p.show()

###############################################################################
# Run the interpolation

interpolated = surface.interpolate(points, radius=12.0)


p = pv.Plotter()
p.add_mesh(points, point_size=30.0, render_points_as_spheres=True)
p.add_mesh(interpolated, scalars="val")
p.show()
