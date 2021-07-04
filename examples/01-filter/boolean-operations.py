"""
.. _boolean_example:

Boolean Operations
~~~~~~~~~~~~~~~~~~

Perform boolean operations with closed (manifold) surfaces.

Boolean/topological operations (intersect, union, difference) methods
are implemented for :class:`pyvista.PolyData` mesh types only and are
accessible directly from any :class:`pyvista.PolyData` mesh. Check out
:class:`pyvista.PolyDataFilters` and take a look at the following
filters:

* :func:`pyvista.PolyDataFilters.boolean_difference`
* :func:`pyvista.PolyDataFilters.boolean_union`
* :func:`pyvista.PolyDataFilters.boolean_intersection`

Essentially, boolean union, difference, and intersection are all the
same operation. Just different parts of the objects are kept at the
end.

The ``-`` operator can be used between any two :class:`pyvista.PolyData`
meshes in PyVista to cut the first mesh by the second.

.. note::
   For merging, the ``+`` operator can be used between any two meshes
   in PyVista which simply calls the ``.merge()`` filter to combine
   any two meshes.  This is difference than ``boolean_union`` as it
   simply adds the two meshes together without operating on them.

"""

# sphinx_gallery_thumbnail_number = 6
import pyvista as pv
import numpy as np

def make_cube():
    x = np.linspace(-0.5, 0.5, 25)
    grid = pv.StructuredGrid(*np.meshgrid(x, x, x))
    return grid.extract_surface().triangulate()

# Create to example PolyData meshes for boolean operations
sphere = pv.Sphere(radius=0.65, center=(0, 0, 0))
cube = make_cube()

p = pv.Plotter()
p.add_mesh(sphere, color="yellow", opacity=0.5, show_edges=True)
p.add_mesh(cube, color="royalblue", opacity=0.5, show_edges=True)
p.show()


###############################################################################
# Boolean Union
# +++++++++++++
#
# Perform a boolean union of ``A`` and ``B``.
# :func:`pyvista.PolyDataFilters.boolean_union` filter.
#
# The union of two manifold meshes ``A`` and ``B`` is the mesh
# which is in ``A``, in ``B``, or in both ``A`` and ``B``.
#
# Order of operations does not matter for boolean union.

add = sphere.boolean_union(cube)
add.plot(opacity=0.5, color=True, show_edges=True)


###############################################################################
# Boolean Difference
# ++++++++++++++++++
#
# Perform a boolean difference of ``A`` and ``B``.
# :func:`pyvista.PolyDataFilters.boolean_difference` filter or the
# ``-`` operator since both meshes are :class:`pyvista.PolyData`.
#
# The difference of two manifold meshes ``A`` and ``B`` is the volume
# of the mesh in ``A`` not belonging to ``B``.
#
# Order of operations matters for boolean cut.

cut = cube - sphere

p = pv.Plotter()
p.add_mesh(cut, opacity=0.5, show_edges=True, color=True)
p.show()


###############################################################################
# Boolean Intersection
# ++++++++++++++++++++
#
# Perform a boolean intersection of ``A`` and ``B``.
# :func:`pyvista.PolyDataFilters.boolean_intersection` filter.
#
# The intersection of two manifold meshes ``A`` and ``B`` is the mesh
# which is the volume of ``A`` that is also in ``B``.
#
# Order of operations does not matter for intersection.

intersect = sphere.boolean_intersection(cube)

p = pv.Plotter()
p.add_mesh(intersect, opacity=0.5, show_edges=True, color=True)
p.show()

