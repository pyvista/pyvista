"""
.. _glyph_table_example:

Table of Glyphs
~~~~~~~~~~~~~~~

``vtk`` supports tables of glyphs from which glyphs are looked
up. This example demonstrates this functionality.
"""

from __future__ import annotations

import numpy as np

import pyvista as pv

# %%
#
# We can allow tables of glyphs in a backward-compatible way by
# allowing a sequence of geometries as well as single (scalar)
# geometries to be passed as the ``geom`` kwarg of :func:`pyvista.DataSetFilters.glyph`.
# An ``indices`` optional keyword specifies the index of each glyph geometry in
# the table, and it has to be the same length as ``geom`` if specified. If it is
# absent a default value of ``range(len(geom))`` is assumed.

# sphinx_gallery_start_ignore
# interactive plot has wrong lighting
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# get dataset for the glyphs: supertoroids in xy plane
# use N random kinds of toroids over a mesh with 27 points
N = 5
values = np.arange(N)  # values for scalars to look up glyphs by


# taken from:
# rng = np.random.default_rng()
# params = rng.uniform(0.5, 2, size=(N, 2))  # (n1, n2) parameters for the toroids
params = np.array(
    [
        [1.56821334, 0.99649769],
        [1.08247844, 1.83758874],
        [1.49598881, 0.83495047],
        [1.52442129, 0.89600688],
        [1.92212387, 0.78096621],
    ],
)

geoms = [pv.ParametricSuperToroid(n1=n1, n2=n2) for n1, n2 in params]

# get dataset where to put glyphs
x, y, z = np.mgrid[:3.0, :3.0, :3.0]
mesh = pv.StructuredGrid(x, y, z)

# add random scalars
# rng_int = rng.integers(0, N, size=x.size)
rng_int = np.array(
    [4, 1, 2, 0, 4, 0, 1, 4, 3, 1, 1, 3, 3, 4, 3, 4, 4, 3, 3, 2, 2, 1, 1, 1, 2, 0, 3],
)
mesh.point_data['scalars'] = rng_int

# construct the glyphs on top of the mesh; don't scale by scalars now
glyphs = mesh.glyph(geom=geoms, indices=values, scale=False, factor=0.3, rng=(0, N - 1))

# create plotter and add our glyphs with some nontrivial lighting
pl = pv.Plotter()
pl.add_mesh(glyphs, specular=1, specular_power=15, smooth_shading=True, show_scalar_bar=False)
pl.show()
# %%
# .. tags:: filter
