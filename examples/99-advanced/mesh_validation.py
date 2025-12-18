"""
.. _mesh_validation_example:

Mesh Validation
~~~~~~~~~~~~~~~
This example demonstrates how to evaluate the validity of a mesh for use with VTK using
:meth:`~pyvista.DataSet.validate_mesh`.

"""

from __future__ import annotations

import pyvista as pv
from pyvista.examples import plot_cell

# %%
# Non-convex cells
# ----------------
# Many VTK algorithms assume that cells are convex. This can result in incorrect outputs
# and may also affect rendering. For example, let's create :class:`~pyvista.PolyData`
# with a concave :attr:`~pyvista.CellType.QUAD` cell.

points = [
    [-0.5, -1.0, 0.0],
    [0.0, -0.3, 0.0],
    [1.0, 0.0, 0.0],
    [-0.5, 0.0, 0.0],
]
faces = [4, 0, 1, 2, 3]
quad = pv.PolyData(points, faces)

# %%
# Use :meth:`~pyvista.DataSet.validate_mesh` to show that the cell is not convex.
report = quad.validate_mesh()
assert not report.is_valid
assert report.issues == ('non_convex',)

# %%
# If we plot the cell, we can see that the concave cell is incorrectly rendered as though it's
# convex even though it is not.
plot_cell(quad, 'xy')

# %%
# To address the convexity issue, we can :meth:`~pyvista.PolyDataFilters.triangulate` the mesh.
# The mesh is now valid and renders correctly.
triangles = quad.triangulate()
report = triangles.validate_mesh()
assert report.is_valid
plot_cell(triangles, 'xy')

# %%
# Cells with inverted faces
# -------------------------
# Cells with inverted faces can result in incorrect geometric computations such as
# cell volume or centroid. To demonstrate this, we first create a valid
# :attr:`~pyvista.CellType.POLYHEDRON` cell similar to the
# :func:`~pyvista.examples.cells.Polyhedron` example cell.

points = [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]]
cells = [4, 3, 0, 2, 1, 3, 0, 1, 3, 3, 0, 3, 2, 3, 1, 2, 3]
cells = [len(cells), *cells.copy()]
polyhedron = pv.UnstructuredGrid(cells, [pv.CellType.POLYHEDRON], points)

# %%
# Plot the cell and show its normals. Since all points have counter-clockwise traversal, the
# normals all point outward and the cell is valid.
report = polyhedron.validate_mesh()
assert report.is_valid
plot_cell(polyhedron, show_normals=True)

# %%
# Now swap two points in the polyhedron's connectivity to generate an otherwise identical
# polyhedron with a single incorrectly oriented face.
index1 = 3  # index of first point ID of first face
index2 = index1 + 1  # index of second point ID of first face
point_id1 = cells[index1]
cells[index1] = cells[index2]
cells[index2] = point_id1

invalid_polyhedron = pv.UnstructuredGrid(cells, [pv.CellType.POLYHEDRON], points)

# %%
# The cell is now invalid, and the bottom face is incorrectly oriented with its normal pointing
# inward.
report = invalid_polyhedron.validate_mesh()
assert not report.is_valid
plot_cell(invalid_polyhedron, show_normals=True)

# %%
# If we review the issues, we see that `two` issues are reported instead of only one.
assert report.issues == ('non_convex', 'inverted_faces')

# %%
# The ``'inverted_faces'`` issue is accurate, but the ``'non_convex'`` issue
# is a false-positive, since the only real problem is with the face orientation. But since the face
# orientation is wrong, it's no longer possible for the mesh validation to accurately determine the
# cell convexity. This can sometimes make identifying the core issue with a cell challenging.
#
# Now let's compare the centroid of the valid and invalid cells using
# :meth:`~pyvista.DataObjectFilters.cell_centers`. The computed centroids differ, demonstrating
# the need to have valid cells when using filters that depend on geometric properties.
valid_centroid = polyhedron.cell_centers().points[0].tolist()
print(valid_centroid)
invalid_centroid = invalid_polyhedron.cell_centers().points[0].tolist()
print(invalid_centroid)
assert valid_centroid != invalid_centroid


# %%
# Meshes with unused points
# -------------------------
# Unused points are points not associated with any cells. These points are not processed
# consistently by filters and are often ignored or removed. To demonstrate this, create an
# :class:`~pyvista.UnstructuredGrid` with a single unused point.
grid = pv.UnstructuredGrid()
grid.points = [[0.0, 0.0, 0.0]]
assert grid.n_points == 1
assert grid.n_cells == 0

# %%
# This mesh is not considered valid.
report = grid.validate_mesh()
assert not report.is_valid
assert report.issues == ('unused_points',)

# %%
# Use :meth:`~pyvista.DataSetFilters.extract_geometry` on the grid and observe that the
# unused point is removed.
poly = grid.extract_geometry()
assert poly.n_points == 0
assert poly.n_cells == 0

# %%
# To remedy this, it is recommended to always associate individual points with a
# :attr:`~pyvista.CellType.VERTEX` cell. E.g.:
points = [[0.0, 0.0, 0.0]]
cells = [1, 0]
celltypes = [pv.CellType.VERTEX]
grid = pv.UnstructuredGrid(cells, celltypes, points)
assert grid.n_points == 1
assert grid.n_cells == 1

# %%
# This time, the point is properly processed by the filter and is retained.
poly = grid.extract_geometry()
assert poly.n_points == 1
assert poly.n_cells == 1

# %%
# This mesh is also now considered valid.
report = grid.validate_mesh()
assert report.is_valid
assert report.issues is None
