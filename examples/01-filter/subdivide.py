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
    position=(-0.02788, 0.1929, 0.4334),
    focal_point=(-0.05326, 0.08881, -9.017e-05),
    viewup=(-0.1017, 0.9686, -0.2267),
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
    show_edges=True,
    color=True,
    cpos=cpos,
)
# %%
# .. tags:: filter
