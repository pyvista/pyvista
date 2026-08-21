"""
.. _interpolate_example:

Detailed Interpolating Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses :func:`pyvista.DataSetFilters.interpolate`.
:func:`pyvista.DataObjectFilters.sample` is similar, and the two
methods are compared in :ref:`interpolate_sample_example`.

Interpolate one mesh's point/cell arrays onto another mesh's nodes using a
Gaussian Kernel.
"""

# sphinx_gallery_thumbnail_number = 4
import pyvista as pv
from pyvista import examples

# %%
# Simple Surface Interpolation
# ++++++++++++++++++++++++++++
# Resample the points' arrays onto a surface

# Download sample data
surface = examples.download_saddle_surface()
points = examples.download_sparse_points()

pl = pv.Plotter()
pl.add_mesh(points, scalars='val', point_size=30.0, render_points_as_spheres=True)
pl.add_mesh(surface)
pl.show()

# %%
# Run the interpolation

interpolated = surface.interpolate(points, radius=12.0)


pl = pv.Plotter()
pl.add_mesh(points, scalars='val', point_size=30.0, render_points_as_spheres=True)
pl.add_mesh(interpolated, scalars='val')
pl.show()


# %%
# Complex Interpolation
# +++++++++++++++++++++
# In this example, we will in interpolate sparse points in 3D space into a
# volume. These data are from temperature probes in the subsurface and the goal
# is to create an approximate 3D model of the temperature field in the
# subsurface.
#
# This approach is a great for back-of-the-hand estimations but pales in
# comparison to kriging

# Download the sparse data
probes = examples.download_thermal_probes()

# %%
# Create the interpolation grid around the sparse data
grid = pv.ImageData()
grid.origin = (329700, 4252600, -2700)
grid.spacing = (250, 250, 50)
grid.dimensions = (60, 75, 100)

# %%
dargs = dict(cmap='coolwarm', clim=[0, 300], scalars='temperature (C)')
cpos = pv.CameraPosition(
    position=(364300.0, 4285000.0, 14090.0),
    focal_point=(337700.0, 4261000.0, -637.1),
    viewup=(-0.2963, -0.2384, 0.9249),
)

pl = pv.Plotter()
pl.add_mesh(grid.outline(), color='k')
pl.add_mesh(probes, render_points_as_spheres=True, **dargs)
pl.show(cpos=cpos)


# %%
# Run an interpolation
interp = grid.interpolate(probes, radius=15000, sharpness=10, strategy='mask_points')

# %%
# Visualize the results

# sphinx_gallery_start_ignore
# volume rendering does not work in interactive plots currently
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

vol_opac = [0, 0, 0.2, 0.2, 0.5, 0.5]

pl = pv.Plotter(shape=(1, 2), window_size=[1024 * 3, 768 * 2])
pl.add_volume(interp, opacity=vol_opac, **dargs)
pl.add_mesh(probes, render_points_as_spheres=True, point_size=10, **dargs)
pl.subplot(0, 1)
pl.add_mesh(interp.contour(5), opacity=0.5, **dargs)
pl.add_mesh(probes, render_points_as_spheres=True, point_size=10, **dargs)
pl.link_views()
pl.show(cpos=cpos)
# %%
# .. tags:: filter
