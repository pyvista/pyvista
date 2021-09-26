"""
.. _platonic_example:

Platonic Solids
~~~~~~~~~~~~~~~

PyVista wraps the ``vtk.vtkPlatonicSolid`` filter as
:func:`pyvista.PlatonicSolid`.
"""
import numpy as np
import pyvista as pv
from pyvista import examples

###############################################################################
# We can either use the generic :func:`PlatonicSolid() <pyvista.PlatonicSolid>`
# and specify the different kinds of solids to generate, or we can use the thin
# wrappers
#
#     * :func:`pyvista.Tetrahedron`
#     * :func:`pyvista.Octahedron`
#     * :func:`pyvista.Dodecahedron`
#     * :func:`pyvista.Icosahedron`
#
# (:func:`PlatonicSolid() <pyvista.PlatonicSolid>` can also return a cube, but
# PyVista's existing :func:`pyvista.Cube` helper isn't based on the
# ``vtkPlatonicSolid`` filter.)
#
# Let's generate all the Platonic solids, along with the :func:`teapotahedron
# <pyvista.examples.downloads.download_teapot>`.

kinds = [
    'tetrahedron',
    'cube',
    'octahedron',
    'dodecahedron',
    'icosahedron',
]
centers = [
    ( 0, 1, 0),
    ( 0, 0, 0),
    ( 0, 2, 0),
    (-1, 0, 0),
    (-1, 2, 0),
]

solids = [
    pv.PlatonicSolid(kind, radius=0.4, center=center)
    for kind, center in zip(kinds, centers)
]

# download and align teapotahedron
teapot = examples.download_teapot()
teapot.rotate_x(90)
teapot.rotate_z(-45)
teapot.scale(0.16)
teapot.points += np.array([-1, 1, 0]) - teapot.center
solids.append(teapot)

###############################################################################
# Now let's plot them all.

p = pv.Plotter()
for ind, solid in enumerate(solids):
    # only use smooth shading for the teapot
    smooth_shading = ind == len(solids) - 1
    p.add_mesh(solid, color='silver', smooth_shading=smooth_shading,
               specular=1.0, specular_power=10)
p.view_vector((5.0, 2, 3))
p.add_floor('-z', lighting=True, color='navy', pad=1.0)
p.show()

###############################################################################
# The conventional Platonic solids come with cell scalars that index each face
# of the solids (unlike the output of :func:`pyvista.Cube`).
