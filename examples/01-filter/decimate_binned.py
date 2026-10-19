"""
.. _decimate_binned_example:

Binned Decimation
~~~~~~~~~~~~~~~~~

Quickly thin a large surface with :func:`pyvista.PolyDataFilters.decimate_binned`.

The filter splits the bounds of the mesh into a grid of bins and merges the points
in each bin into one. It does not preserve topology, but it is orders of magnitude
faster than :func:`pyvista.PolyDataFilters.decimate` and
:func:`pyvista.PolyDataFilters.decimate_pro`, which makes it a good fit for previews
of very large surfaces.

"""

# sphinx_gallery_thumbnail_number = 2
import time

import pyvista as pv
from pyvista import examples

mesh = examples.download_armadillo()
cpos = pv.CameraPosition(
    position=(161.5, 82.1, -330.2), focal_point=(-4.3, 24.5, -1.6), viewup=(-0.1, 1, 0.12)
)
mesh

# %%
# Decimate the mesh to roughly 5,000 points.

decimated = mesh.decimate_binned(5_000)
decimated.plot(cpos=cpos, show_edges=True)

# %%
# Compare the Filters
# -------------------
# Time each of the three decimation filters on the same reduction.

target = 5_000
reduction = 1 - target / mesh.n_points
filters = {
    'decimate': lambda: mesh.decimate(reduction),
    'decimate_pro': lambda: mesh.decimate_pro(reduction),
    'decimate_binned': lambda: mesh.decimate_binned(target),
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

pv.plot_compare(results, cpos=cpos, window_size=[1200, 500], show_edges=True)

# %%
# .. tags:: filter
