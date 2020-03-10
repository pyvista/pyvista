"""
Boolean Operations
~~~~~~~~~~~~~~~~~~

Perform boolean operations with closed surfaces (intersect, cut, etc.).

Boolean/topological operations (intersect, cut, etc.) methods are implemented
for :class:`pyvista.PolyData` mesh types only and are accessible directly from
any :class:`pyvista.PolyData` mesh. Check out :class:`pyvista.PolyDataFilters`
and take a look at the following filters:

* :func:`pyvista.PolyDataFilters.boolean_add`
* :func:`pyvista.PolyDataFilters.boolean_cut`
* :func:`pyvista.PolyDataFilters.boolean_difference`
* :func:`pyvista.PolyDataFilters.boolean_union`

For merging, the ``+`` operator can be used between any two meshes in PyVista
which simply calls the ``.merge()`` filter to combine any two meshes.
Similarly, the ``-`` operator can be used between any two :class:`pyvista.PolyData`
meshes in PyVista to cut the first mesh by the second.
"""

# sphinx_gallery_thumbnail_number = 6
import pyvista as pv
import numpy as np

def make_cube():
    x = np.linspace(-0.5, 0.5, 25)
    grid = pv.StructuredGrid(*np.meshgrid(x, x, x))
    return grid.extract_surface().triangulate()

# Create to examplee PolyData meshes for boolean operations
sphere = pv.Sphere(radius=0.65, center=(0, 0, 0))
cube = make_cube()

p = pv.Plotter()
p.add_mesh(sphere, color="yellow", opacity=0.5, show_edges=True)
p.add_mesh(cube, color="royalblue", opacity=0.5, show_edges=True)
p.show()

###############################################################################
# Boolean Add
# +++++++++++
#
# Add all of the two meshes together using the
# :func:`pyvista.PolyDataFilters.boolean_add` filter or the ``+`` operator.
#
# Order of operations does not matter for boolean add as the entirety of both
# meshes are appended together.

add = sphere + cube
add.plot(opacity=0.5, color=True, show_edges=True)


###############################################################################
# Boolean Cut
# +++++++++++
#
# Perform a boolean cut of ``a`` using ``b`` with the
# :func:`pyvista.PolyDataFilters.boolean_cut` filter or the ``-`` operator
# since both meshes are :class:`pyvista.PolyData`.
#
# Order of operations does not matter for boolean cut.

cut = cube - sphere

p = pv.Plotter()
p.add_mesh(cut, opacity=0.5, show_edges=True, color=True)
p.show()


###############################################################################
# Boolean Difference
# ++++++++++++++++++
#
# Combine two meshes and retains only the volume in common between the meshes
# using the :func:`pyvista.PolyDataFilters.boolean_difference` method.
#
# Note that the order of operations for a boolean difference will affect the
# results.

diff = sphere.boolean_difference(cube)

p = pv.Plotter()
p.add_mesh(diff, opacity=0.5, show_edges=True, color=True)
p.show()


###############################################################################

diff = cube.boolean_difference(sphere)

p = pv.Plotter()
p.add_mesh(diff, opacity=0.5, show_edges=True, color=True)
p.show()

###############################################################################
# Boolean Union
# +++++++++++++
#
# Combine two meshes and attempts to create a manifold mesh using the
# :func:`pyvista.PolyDataFilters.boolean_union` method.
#
# Order of operations does not matter for boolean union.

union = sphere.boolean_union(cube)

p = pv.Plotter()
p.add_mesh(union,  opacity=0.5, show_edges=True, color=True)
p.show()
