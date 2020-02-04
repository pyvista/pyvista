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

# sphinx_gallery_thumbnail_number = 4
import pyvista as pv

# Create to examplee PolyData meshes for boolean operations
a = pv.Sphere(radius=1, center=(0, 0, 0))
b = pv.Sphere(radius=1, center=(0.5, 0, 0))

cpos = [(-1.1141408809624171, -4.055003246648033, 2.5275085053563373),
 (0.30835059413057386, -0.02838894321931945, -0.012326840224821223),
 (0.1877813462433568, 0.47569601812030493, 0.8593319872712285)]

display_args = dict(opacity=0.5, show_edges=True)

p = pv.Plotter()
p.add_mesh(a, color="lightblue", **display_args)
p.add_mesh(b, color="maroon", **display_args)
p.camera_position = cpos
p.show()

###############################################################################
# Boolean Add
# +++++++++++
#
# Add the two meshes together using the :func:`pyvista.PolyDataFilters.boolean_add`
# filter or the ``+`` operator.

c = a + b
c.plot(cpos=cpos, **display_args)


###############################################################################
# Boolean Cut
# +++++++++++
#
# Perform a boolean cut of ``a`` using ``b`` with the
# :func:`pyvista.PolyDataFilters.boolean_cut` filter or the ``-`` operator
# since both meshes are :class:`pyvista.PolyData`.

c = a - b

p = pv.Plotter()
p.add_mesh(a, color="lightblue", **display_args)
p.add_mesh(b, color="maroon", **display_args)
p.add_mesh(c, color="brown", **display_args)
p.camera_position = cpos
p.show()

###############################################################################
# Boolean Difference
# ++++++++++++++++++
#
# Combine two meshes and retains only the volume in common between the meshes
# using the :func:`pyvista.PolyDataFilters.boolean_difference` method.
c = b.boolean_difference(a)

p = pv.Plotter()
p.add_mesh(c, color="brown", **display_args)
p.camera_position = cpos
p.show()

###############################################################################
# Boolean Union
# +++++++++++++
#
# Combine two meshes and attempts to create a manifold mesh using the
# :func:`pyvista.PolyDataFilters.boolean_union` method.

c = a.boolean_union(b)

p = pv.Plotter()
p.add_mesh(c, color="brown", **display_args)
p.camera_position = cpos
p.show()
