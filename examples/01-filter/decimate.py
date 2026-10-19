"""
.. _decimate_example:

Decimation
~~~~~~~~~~

Decimate a mesh.

"""

# sphinx_gallery_thumbnail_number = 4
import time

import numpy as np
import pyvista as pv
from pyvista import examples

mesh = examples.download_face()

# Define a camera position that shows this mesh properly
cpos = pv.CameraPosition(
    position=(0.4, -0.07, -0.31), focal_point=(0.05, -0.13, -0.06), viewup=(-0.1, 1, 0.08)
)
plot_kwargs = dict(show_edges=True, color=True)

# Preview the mesh
mesh.plot(cpos=cpos, **plot_kwargs)

# %%
# Now let's define a target reduction and compare the
# :func:`pyvista.PolyDataFilters.decimate` and
# :func:`pyvista.PolyDataFilters.decimate_pro` filters.
target_reduction = 0.7
print(f'Reducing {target_reduction * 100.0} percent out of the original mesh')

# %%
decimated = mesh.decimate(target_reduction)

decimated.plot(cpos=cpos, **plot_kwargs)


# %%
pro_decimated = mesh.decimate_pro(target_reduction, preserve_topology=True)

pro_decimated.plot(cpos=cpos, **plot_kwargs)


# %%
# Side by side comparison:

# sphinx_gallery_start_ignore
# text missing in interactive
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

datasets = {
    'Input': mesh,
    'Decimated': decimated,
    'Pro Decimated': pro_decimated,
}

pv.plot_compare(
    datasets,
    label_size=24,
    cpos=cpos,
    **plot_kwargs,
)

# %%
# Decimate PolyLine Mesh
# ----------------------
#
# Generate a fairly slow spiral polyline mesh.

n_points = 100
n_rotations = 5
phi = np.linspace(0, 2 * np.pi * n_rotations, n_points)
ratio = 1.1
r = ratio ** (phi)

points = np.zeros((n_points, 3))
points[:, 0] = r * np.cos(phi)
points[:, 1] = r * np.sin(phi)

spiral = pv.PolyData(points, lines=np.append([n_points], np.arange(n_points)))

# %%
# Construct a reusable plotting function for future use.


def compare_decimation(spiral, decimated):
    pl = pv.Plotter()
    pl.add_mesh(spiral, line_width=5, color='r', label='Original')
    pl.add_mesh(decimated, line_width=3, color='k', label='Decimated')
    pl.view_xy()
    pl.add_legend(face='line', size=(0.25, 0.25))


# %%
# Decimate using :func:`pyvista.PolyDataFilters.decimate_polyline` filter by
# target of 50%.

decimated = spiral.decimate_polyline(0.5)
print(f'Original # of points:  {spiral.n_points}')
print(f'Decimated # of points: {decimated.n_points}')

# %%
# The decimation looks OK at this level of reduction.

compare_decimation(spiral, decimated)

# %%
# Using a larger level of reduction, 80%, leads to a much coarser level of
# representation.

decimated = spiral.decimate_polyline(0.8)
print(f'Original # of points:  {spiral.n_points}')
print(f'Decimated # of points: {decimated.n_points}')

# %%
# The structure of the inner part of the spiral is completely
# lost.

compare_decimation(spiral, decimated)

# %%
# To avoid errors of quickly changing features, use the ``maximum_error``
# parameter. It is in units of fraction of the largest length of the
# bounding box. Note that it limits the level of reduction achieved.

decimated = spiral.decimate_polyline(0.8, maximum_error=0.5)
print(f'Original # of points:  {spiral.n_points}')
print(f'Decimated # of points: {decimated.n_points}')

# %%
# The structure of the inner part of the spiral is captured adequately.

compare_decimation(spiral, decimated)

# %%
# Binned Decimation
# -----------------
# :func:`pyvista.PolyDataFilters.decimate_binned` splits the bounds of a mesh into
# a grid of bins and merges the points in each bin into one. It does not preserve
# topology, but it is orders of magnitude faster than the filters above, which makes
# it a good fit for previews of very large surfaces.
#
# Decimate a larger mesh to roughly 5,000 points.

armadillo = examples.download_armadillo()
armadillo_cpos = pv.CameraPosition(
    position=(161.5, 82.1, -330.2), focal_point=(-4.3, 24.5, -1.6), viewup=(-0.1, 1, 0.12)
)
binned = armadillo.decimate_binned(5_000)
binned.plot(cpos=armadillo_cpos, show_edges=True)

# %%
# Time each of the three surface decimation filters on the same reduction.

target = 5_000
reduction = 1 - target / armadillo.n_points
filters = {
    'decimate': lambda: armadillo.decimate(reduction),
    'decimate_pro': lambda: armadillo.decimate_pro(reduction),
    'decimate_binned': lambda: armadillo.decimate_binned(target),
}

results = {}
for name, run in filters.items():
    start = time.perf_counter()
    results[name] = run()
    print(f'{name}: {time.perf_counter() - start:.3f} s')

# %%
# Plot the results side by side.

# sphinx_gallery_start_ignore
# labels are missing in interactive plots
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pv.plot_compare(results, cpos=armadillo_cpos, window_size=[1200, 500], show_edges=True)

# %%
# .. tags:: filter
