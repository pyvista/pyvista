"""
.. _mesh_validation_example:

Mesh Validation
~~~~~~~~~~~~~~~
This example demonstrates how to evaluate the validity of a mesh for use with VTK using
:meth:`~pyvista.DataSetFilters.cell_validator` and :meth:`~pyvista.DataSet.validate_mesh`.

"""

from __future__ import annotations

import pyvista as pv
from pyvista.examples import plot_cell

# %%
# Cell Convexity
# --------------
# VTK assumes that cells are convex. If cells are non-convex, they may not be rendered as
# expected. For example, let's create :class:`~pyvista.PolyData` with a concave
# :attr:`~pyvista.CellType.QUAD` cell.

points = [
    [-0.5, -1.0, 0.0],
    [0.0, -0.3, 0.0],
    [1.0, 0.0, 0.0],
    [-0.5, 0.0, 0.0],
]
faces = [4, 0, 1, 2, 3]
quad = pv.PolyData(points, faces)

# %%
# Use :meth:`~pyvista.DataSetFilters.validate_mesh` to show that the cell is not convex.
report = quad.validate_mesh()
print(report.is_valid)
print(report.issues)

# %%
# If we plot the cell, we can see that the concave cell is incorrectly rendered as though it's
# convex even though it is not.
plot_cell(quad, 'xy')

# %%
# To fix the convexity issue, we can :meth:`~pyvista.PolyDataFilters.triangulate` the mesh.
triangles = quad.triangulate()
report = triangles.validate_mesh()
print(report.is_valid)
print(report.issues)
plot_cell(triangles, 'xy')
