"""
.. _streamlines_2D_example:

2D Streamlines
~~~~~~~~~~~~~~

Integrate a vector field to generate streamlines on a 2D surface.
"""

# sphinx_gallery_thumbnail_number = 3

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

# %%
# This example generates streamlines of flow around a cylinder in cross flow.
from __future__ import annotations

import pyvista as pv
from pyvista import examples

# %%
# The data is multiblock with the fluid data as the first block.
# The data lies in the `xy` plane, i.e. `z=0`, with no `z` velocity.

mesh = examples.download_cylinder_crossflow()
fluid_mesh = mesh[0]
print(fluid_mesh)

# %%
# The default behavior of the :func:`streamlines()
# <pyvista.DataSetFilters.streamlines>` filter is to use a 3D sphere source as
# the seed points.  This often will not generate any seed points on the 2D
# plane of interest.  Instead, a single streamline can be generated using the
# ``start_position`` argument. The ``surface_streamlines=True`` argument is
# also needed if the dataset has nonzero normal velocity component.  This is
# not the case in this dataset.

one_streamline = fluid_mesh.streamlines(
    start_position=(0.0, 0.4, 0.0),
    max_length=100.0,
    compute_vorticity=False,  # vorticity already exists in dataset
)

clim = [0, 20]
camera_position = pv.CameraPosition(
    position=(7, 0, 20.0), focal_point=(7, 0.0, 0.0), viewup=(0.0, 1.0, 0.0)
)

pl = pv.Plotter()
for i in range(1, len(mesh)):
    pl.add_mesh(mesh[i], color='k')
pl.add_mesh(one_streamline.tube(radius=0.05), scalars='vorticity_mag', clim=clim)
pl.view_xy()
pl.show(cpos=camera_position)

# %%
# To generate multiple streamlines, a line source can be used with the ``pointa``
# and ``pointb`` parameters.

line_streamlines = fluid_mesh.streamlines(
    pointa=(0, -5, 0),
    pointb=(0, 5, 0),
    n_points=25,
    max_length=100.0,
    compute_vorticity=False,  # vorticity already exists in dataset
)

pl = pv.Plotter()
for i in range(1, len(mesh)):
    pl.add_mesh(mesh[i], color='k')
pl.add_mesh(line_streamlines.tube(radius=0.05), scalars='vorticity_mag', clim=clim)
pl.view_xy()
pl.show(cpos=camera_position)

# %%
# The behavior immediately downstream of the cylinder is still not apparent
# using streamlines at the inlet.
#
# Another method is to use :func:`streamlines_evenly_spaced_2D()
# <pyvista.DataSetFilters.streamlines_evenly_spaced_2D>`.
# This filter only works with 2D data that lies on the xy plane. This method
# can quickly run of memory, so particular attention must be paid to the input
# parameters.  The defaults are in cell length units.

line_streamlines = fluid_mesh.streamlines_evenly_spaced_2D(
    start_position=(4, 0.1, 0.0),
    separating_distance=3,
    separating_distance_ratio=0.2,
    compute_vorticity=False,  # vorticity already exists in dataset
)

pl = pv.Plotter()
for i in range(1, len(mesh)):
    pl.add_mesh(mesh[i], color='k')
pl.add_mesh(line_streamlines.tube(radius=0.02), scalars='vorticity_mag', clim=clim)
pl.view_xy()
pl.show(cpos=camera_position)

# %%
# The streamlines are only approximately evenly spaced and capture the
# vortex pair downstream of the cylinder with appropriate choice of
# ``start_position``.
#
# .. tags:: filter
