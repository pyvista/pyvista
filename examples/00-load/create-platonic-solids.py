"""
.. _platonic_example:

Platonic Solids
~~~~~~~~~~~~~~~

PyVista wraps the ``vtk.vtkPlatonicSolidSource`` filter as
:func:`pyvista.PlatonicSolid`.
"""
import numpy as np

import pyvista as pv
from pyvista import examples

###############################################################################
# We can either use the generic :func:`PlatonicSolid() <pyvista.PlatonicSolid>`
# and specify the different kinds of solids to generate, or we can use the thin
# wrappers:
#
#     * :func:`pyvista.Tetrahedron`
#     * :func:`pyvista.Octahedron`
#     * :func:`pyvista.Dodecahedron`
#     * :func:`pyvista.Icosahedron`
#     * :func:`pyvista.Cube` (implemented via a different filter)
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
    (0, 1, 0),
    (0, 0, 0),
    (0, 2, 0),
    (-1, 0, 0),
    (-1, 2, 0),
]

solids = [pv.PlatonicSolid(kind, radius=0.4, center=center) for kind, center in zip(kinds, centers)]

# download and align teapotahedron
teapot = examples.download_teapot()
teapot.rotate_x(90, inplace=True)
teapot.rotate_z(-45, inplace=True)
teapot.scale(0.16, inplace=True)
teapot.points += np.array([-1, 1, 0]) - teapot.center
solids.append(teapot)

###############################################################################
# Now let's plot them all.
#
# .. note::
#    VTK has known issues when rendering shadows on certain window
#    sizes.  Be prepared to experiment with the ``window_size``
#    parameter.  An initial window size of ``(1000, 1000)`` seems to
#    work well, which can be manually resized without issue.


p = pv.Plotter(window_size=[1000, 1000])
for ind, solid in enumerate(solids):
    # only use smooth shading for the teapot
    smooth_shading = ind == len(solids) - 1
    p.add_mesh(
        solid, color='silver', smooth_shading=smooth_shading, specular=1.0, specular_power=10
    )
p.view_vector((5.0, 2, 3))
p.add_floor('-z', lighting=True, color='tan', pad=1.0)
p.enable_shadows()
p.show()

###############################################################################
# The Platonic solids come with cell scalars that index each face of the
# solids.
