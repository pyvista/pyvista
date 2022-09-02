"""
.. _linear_cells_example:

Linear Cells
~~~~~~~~~~~~

This example extends the :ref:`create_unstructured_example` example by
including an explanation of linear VTK cell types and how you can create them in
PyVista.

Linear cells are cells where points only occur at the edges of each
cell. Non-linear cells contain additional points along the edges of the cell.

For more details regarding what a :class:`pyvista.UnstructuredGrid` is, please
see :ref:`point_sets_api`.

"""

import pyvista as pv

###############################################################################
# Plot an example cell
# ~~~~~~~~~~~~~~~~~~~~
# PyVista contains a simple utility to plot a single cell, which is the
# fundamental unit of each :class:`pyvista.UnstructuredGrid`. For example,
# let's plot a simple :func:`Wedge <pyvista.cells.Wedge`.
#
grid = pv.cells.Wedge()
pv.cells.plot_cell(grid)


###############################################################################
# This linear cell is composed of 6 points.

grid.points


###############################################################################
# The UnstructuredGrid is also composed of a single cell and the point indices
# of that cell are defined in :attr:`cells <pyvista.UnstructuredGrid.cells>`.
#
# .. note::
#    The leading ``6`` is the number of points in the cell.

grid.cells


###############################################################################
# Combine two UnstructuredGrids
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can combine two unstructured grids to create a single unstructured grid
# using the ``+`` operator.
#
# .. note::
#    This is an inefficient way of creating :class:`UnstructuredGrid`
#    objects. To see a more efficient implement ion see
#    :ref:`create_unstructured_example`.

grid_a = pv.cells.Hexahedron()
grid_a.points += [0, 2.5, 0]

grid_b = pv.cells.HexagonalPrism()

combined = grid_b + grid_a

pv.cells.plot_cell(combined, cpos='iso')


###############################################################################
# This example helps to illustrate meaning behind the :attr:`cells
# <pyvista.UnstructuredGrid.cells>` attribute. The first cell, a hexahedron
# contains 8 points and the hexagonal prism contains 12 points. The ``cells``
# attribute shows this along with indices composing each cell.

combined.cells


###############################################################################
# Cell Types
# ~~~~~~~~~~
# PyVista contains the :class:`pv.CellType` enumerator, which contains all the
# available VTK cell types mapped to a Python enumerator. These cell types are
# used when creating cells and also can be used when checking the
# :attr:`celltypes <pyvista.UnstructuredGrid.celltypes>` attribute. For example
# ``combined.celltypes`` contains both the ``pv.CellType.HEXAHEDRON`` and
# ``pv.CellType.HEXAGONAL_PRISM`` cell types.

print(pv.CellType.HEXAHEDRON, pv.CellType.HEXAGONAL_PRISM)
(pv.CellType.HEXAHEDRON, pv.CellType.HEXAGONAL_PRISM) == combined.celltypes


###############################################################################
# Create an UnstructuredGrid with a single linear cell
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now that you know the three main inputs of an
# :class:`pyvista.UnstructuredGrid`, it's quite straightforward to create an
# unstructured grid with a one or more cells. If you need to reference point
# ordering or additional, you can either read the source of `cells.py
# <https://github.com/pyvista/pyvista/blob/main/pyvista/cells/cells.py>`_ or
# simply create a cell from the ``pyvista.cells`` module and inspect its attributes.

points = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [0.0, 0.0, 1.60803807],
]
cells = [len(points)] + list(range(len(points)))
pyrmaid = pv.UnstructuredGrid(cells, [pv.CellType.PYRAMID], points)
pv.cells.plot_cell(pyrmaid)


###############################################################################
# Plot all linear cell Types
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's create a ``(4, 4)`` :class:`pyvista.Plotter` and plot all 16 linear
# cells in a single plot.


def add_cell_helper(pl, text, grid, subplot, cpos=None):
    """Add a single cell to a plotter with fancy plotting."""
    pl.subplot(*subplot)
    pl.add_text(text, 'lower_edge', color='k', font_size=10)
    pl.add_mesh(grid, opacity=0.5, color='tan', line_width=5)
    edges = grid.extract_all_edges()
    if edges.n_cells:
        pl.add_mesh(grid.extract_all_edges(), line_width=5, color='k')
    pl.add_points(grid, render_points_as_spheres=True, point_size=20, color='r')
    pl.add_point_labels(
        grid.points,
        range(grid.n_points),
        always_visible=True,
        fill_shape=False,
        margin=0,
        shape_opacity=0.0,
        font_size=20,
        text_color='k',
    )
    if cpos is None:
        pl.camera.azimuth = 20
        pl.camera.elevation = -20
    else:
        pl.camera_position = cpos
    pl.camera.zoom(0.8)


pl = pv.Plotter(shape=(4, 4))
add_cell_helper(pl, 'VERTEX', pv.cells.Vertex(), (0, 0))
add_cell_helper(pl, 'POLY_VERTEX', pv.cells.PolyVertex(), (0, 1))
add_cell_helper(pl, 'LINE', pv.cells.Line(), (0, 2))
add_cell_helper(pl, 'POLY_LINE', pv.cells.PolyLine(), (0, 3))

add_cell_helper(pl, 'TRIANGLE', pv.cells.Triangle(), (1, 0), cpos='xy')
add_cell_helper(
    pl, 'TRIANGLE_STRIP', pv.cells.TriangleStrip().rotate_z(90, inplace=False), (1, 1), cpos='xy'
)
add_cell_helper(pl, 'POLYGON', pv.cells.Polygon(), (1, 2), cpos='xy')
add_cell_helper(pl, 'PIXEL', pv.cells.Pixel(), (1, 3), cpos='xy')

add_cell_helper(pl, 'QUAD', pv.cells.Quadrilateral(), (2, 0))
add_cell_helper(pl, 'TETRA', pv.cells.Tetrahedron(), (2, 1))
add_cell_helper(pl, 'VOXEL', pv.cells.Voxel(), (2, 2))
add_cell_helper(pl, 'HEXAHEDRON', pv.cells.Hexahedron(), (2, 3))

add_cell_helper(pl, 'WEDGE', pv.cells.Wedge(), (3, 0))
add_cell_helper(pl, 'PYRAMID', pv.cells.Pyramid(), (3, 1))
add_cell_helper(pl, 'PENTAGONAL_PRISM', pv.cells.PentagonalPrism(), (3, 2))
add_cell_helper(pl, 'HEXAGONAL_PRISM', pv.cells.HexagonalPrism(), (3, 3))

pl.background_color = 'w'
pl.enable_anti_aliasing()
pl.show()
