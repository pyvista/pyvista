"""
.. _subdivide_example:

Subdivide Cells
~~~~~~~~~~~~~~~

Increase the number of triangles in a single, connected triangular mesh.

The :func:`pyvista.PolyDataFilters.subdivide` filter utilizes three different
subdivision algorithms to subdivide a mesh's cells: `butterfly`, `loop`,
or `linear`.
"""

import pyvista as pv
from pyvista import examples

# %%
# First, let's load a **triangulated** mesh to subdivide. We can use the
# :func:`pyvista.DataObjectFilters.triangulate` filter to ensure the mesh we are
# using is purely triangles.
mesh = examples.download_bunny_coarse().triangulate().clean()

cpos = pv.CameraPosition(
    position=(-0.02788175062966399, 0.19293295656233056, 0.4334449972621349),
    focal_point=(-0.053260899930287015, 0.08881197167521734, -9.016948161029588e-05),
    viewup=(-0.10170607813337212, 0.9686438023715356, -0.22668272496584665),
)

# %%
# Now, lets do a few subdivisions with the mesh and compare the results.
# Below is a helper function which collects the meshes and labels for the
# comparison plot of the three different subdivisions.


def subdivisions(mesh, a, b):
    """Return the original mesh and its subdivisions, one row per subfilter."""
    datasets = pv.MultiBlock()
    for subfilter in ['linear', 'butterfly', 'loop']:
        datasets.append(mesh, 'Original Mesh')
        for n_subdivisions in [a, b]:
            datasets.append(
                mesh.subdivide(n_subdivisions, subfilter=subfilter),
                f'{subfilter} subdivision of {n_subdivisions}',
            )
    return datasets


# %%
# Run the subdivisions for 1 and 3 levels and compare them with
# :func:`~pyvista.plot_compare`. The block names are used as labels.

datasets = subdivisions(mesh, 1, 3)

pv.plot_compare(
    datasets,
    shape=(3, 3),
    dataset_kwargs=dict(show_edges=True, color=True),
    cpos=cpos,
)
# %%
# .. tags:: filter
