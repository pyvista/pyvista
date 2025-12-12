"""Contains a variety of cells to serve as examples.

Functions that create single cell :class:`pyvista.UnstructuredGrid` objects which can
be used to learn about VTK :class:`cell types <pyvista.CellType>`.

"""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import CellType
from pyvista import UnstructuredGrid
from pyvista.core import _vtk_core as _vtk


def plot_cell(grid, cpos=None, *, show_normals: bool = True, **kwargs):
    """Plot a :class:`pyvista.UnstructuredGrid` while displaying cell indices.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Unstructured grid (ideally) containing a single cell.

    cpos : str, optional
        Camera position.

    show_normals : bool, optional
        Show the face normals of the cell. Only applies to 2D or 3D cells.
        Cell faces with correct orientation should have the normal pointing outward.

        .. versionadded:: 0.47

    **kwargs : dict, optional
        Additional keyword arguments when showing. See :func:`pyvista.Plotter.show`.

    Examples
    --------
    Create and plot a single hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.Hexahedron()
    >>> examples.plot_cell(grid)

    """

    def _extract_geometry(cell_):
        if cell_.type == pv.CellType.POLYHEDRON:
            # For Polyhedron, we don't use ``extract_geometry`` because that may alter the face
            # orientation, so we iterate over each face directly instead
            faces = []
            for i in range(cell_.n_faces):
                face = cell_.GetFace(i)
                face_n_points = face.GetNumberOfPoints()
                point_ids = [face.GetPointId(i) for i in range(face_n_points)]
                faces.extend([face_n_points, *point_ids])
            return pv.PolyData(cell_.points, faces)
        return cell_.cast_to_unstructured_grid().extract_geometry()

    pl = pv.Plotter()
    for cell in grid.cell:
        # Use existing grid if it's already a grid with one cell
        cell_as_grid = grid if grid.n_cells == 1 else cell.cast_to_unstructured_grid()
        pl.add_mesh(cell_as_grid, opacity=0.5)
        edges = cell_as_grid.extract_all_edges()
        if edges.n_cells or cell.type in [
            CellType.LINE,
            CellType.POLY_LINE,
            CellType.QUADRATIC_EDGE,
            CellType.CUBIC_LINE,
        ]:
            pl.add_mesh(
                cell_as_grid,
                style='wireframe',
                line_width=10,
                color='k',
                render_lines_as_tubes=True,
            )
        pl.add_points(cell.points, render_points_as_spheres=True, point_size=80, color='r')
        pl.add_point_labels(
            cell.points,
            cell.point_ids,
            always_visible=True,
            fill_shape=False,
            margin=0,
            shape_opacity=0.0,
            font_size=50,
        )

        if show_normals and cell.dimension >= 2:
            surface = _extract_geometry(cell)
            surface = surface.triangulate() if cell.type is CellType.TRIANGLE_STRIP else surface
            pl.add_arrows(
                surface.cell_centers().points,
                surface.cell_normals,
                mag=grid.length / 4,
                color='yellow',
                show_scalar_bar=False,
            )

    pl.enable_anti_aliasing()
    if cpos is None:
        pl.camera.azimuth = 20
        pl.camera.elevation = -20
    else:
        pl.camera_position = cpos
    pl.show(**kwargs)


def Empty() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single empty cell.

    This cell corresponds to the :attr:`pyvista.CellType.EMPTY_CELL` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single empty cell.

    Examples
    --------
    Create a single empty cell.

    >>> from pyvista import examples
    >>> grid = examples.cells.Empty()

    List the grid's cells.

    >>> grid.cells
    array([0])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([], shape=(0, 3), dtype=float64)

    >>> grid.celltypes  # same as pyvista.CellType.EMPTY_CELL
    array([0], dtype=uint8)

    """
    return UnstructuredGrid([0], [CellType.EMPTY_CELL], [])


def Vertex() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Vertex.

    This cell corresponds to the :attr:`pyvista.CellType.VERTEX` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single vertex.

    Examples
    --------
    Create and plot a single vertex.

    >>> from pyvista import examples
    >>> grid = examples.cells.Vertex()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([1, 0])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0., 0., 0.]])

    >>> grid.celltypes  # same as pyvista.CellType.VERTEX
    array([1], dtype=uint8)

    """
    points = [[0.0, 0.0, 0.0]]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.VERTEX], points)


def PolyVertex() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single poly vertex.

    This represents a set of 3D vertices as a single cell.

    This cell corresponds to the :attr:`pyvista.CellType.POLY_VERTEX` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single poly vertex.

    Examples
    --------
    Create and plot a single poly vertex.

    >>> from pyvista import examples
    >>> grid = examples.cells.PolyVertex()
    >>> examples.plot_cell(grid)

    List the grid's cells. This could be any number of points.

    >>> grid.cells
    array([6, 0, 1, 2, 3, 4, 5])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [1. , 0. , 0.4],
                     [0. , 1. , 0.6]])

    >>> grid.celltypes  # same as pyvista.CellType.POLY_VERTEX
    array([2], dtype=uint8)

    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0.4], [0, 1, 0.6]]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.POLY_VERTEX], points)


def Line() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Line.

    This cell corresponds to the :attr:`pyvista.CellType.LINE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single line.

    Examples
    --------
    Create and plot a single line.

    >>> from pyvista import examples
    >>> grid = examples.cells.Line()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([2, 0, 1])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0., 0., 0.],
                     [1., 0., 0.]])

    >>> grid.celltypes  # same as pyvista.CellType.LINE
    array([3], dtype=uint8)

    """
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.LINE], points)


def PolyLine() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single poly line.

    This represents a set of line segments as a single cell.

    This cell corresponds to the :attr:`pyvista.CellType.POLY_LINE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single polyline.

    Examples
    --------
    Create and plot a single polyline.

    >>> from pyvista import examples
    >>> grid = examples.cells.PolyLine()
    >>> examples.plot_cell(grid)

    List the grid's cells. This could be any number of points.

    >>> grid.cells
    array([4, 0, 1, 2, 3])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [0.5, 0. , 0. ],
                     [0.5, 1. , 0. ],
                     [0. , 1. , 0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.POLY_LINE
    array([4], dtype=uint8)

    """
    points = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.POLY_LINE], points)


def Triangle() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Triangle.

    This cell corresponds to the :attr:`pyvista.CellType.TRIANGLE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single triangle.

    Examples
    --------
    Create and plot a single triangle.

    >>> from pyvista import examples
    >>> grid = examples.cells.Triangle()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([3, 0, 1, 2])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[ 0.5       , -0.28867513,  0.        ],
                     [ 0.        ,  0.57735027,  0.        ],
                     [-0.5       , -0.28867513,  0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.TRIANGLE
    array([5], dtype=uint8)

    """
    R33 = np.sqrt(3) / 3
    points = [[0.5, -0.5 * R33, 0.0], [0.0, R33, 0.0], [-0.5, -0.5 * R33, 0.0]]

    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.TRIANGLE], points)


def TriangleStrip() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single triangle strip.

    This cell corresponds to the :attr:`pyvista.CellType.TRIANGLE_STRIP` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single triangle strip.

    Examples
    --------
    Create and plot a single triangle strip.

    >>> from pyvista import examples
    >>> grid = examples.cells.TriangleStrip()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([8, 0, 1, 2, 3, 4, 5, 6, 7])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[1., 0., 0.],
                     [0., 0., 0.],
                     [1., 1., 0.],
                     [0., 1., 0.],
                     [1., 2., 0.],
                     [0., 2., 0.],
                     [1., 3., 0.],
                     [0., 3., 0.]])

    >>> grid.celltypes  # same as pyvista.CellType.TRIANGLE_STRIP
    array([6], dtype=uint8)

    """
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
        [1.0, 3.0, 0.0],
    ]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.TRIANGLE_STRIP], points)


def Polygon() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single polygon.

    This cell corresponds to the :attr:`pyvista.CellType.POLYGON` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single polygon.

    Examples
    --------
    Create and plot a single polygon.

    >>> from pyvista import examples
    >>> grid = examples.cells.Polygon()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([6, 0, 1, 2, 3, 4, 5])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[ 0. ,  0. ,  0. ],
                     [ 1. , -0.1,  0. ],
                     [ 0.8,  0.5,  0. ],
                     [ 1. ,  1. ,  0. ],
                     [ 0.6,  1.2,  0. ],
                     [ 0. ,  0.8,  0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.POLYGON
    array([7], dtype=uint8)

    """
    points = [[0, 0, 0], [1, -0.1, 0], [0.8, 0.5, 0], [1, 1, 0], [0.6, 1.2, 0], [0, 0.8, 0]]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.POLYGON], points)


def Polyhedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single polyhedron.

    This cell corresponds to the :attr:`pyvista.CellType.POLYHEDRON` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single polyhedron.

    Examples
    --------
    Create and plot a single polyhedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.Polyhedron()
    >>> examples.plot_cell(grid)

    A polyhedron is defined by it's faces. List the grid's faces.

    >>> grid.get_cell(0).faces  # doctest: +ELLIPSIS
    [Cell...
    ..., Cell...
    ..., Cell...
    ...]

    >>> grid.celltypes  # same as pyvista.CellType.POLYHEDRON
    array([42], dtype=uint8)

    """
    points = [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]]
    cells = [4, 3, 0, 2, 1, 3, 0, 1, 3, 3, 0, 3, 2, 3, 1, 2, 3]
    cells = [len(cells), *cells]
    return UnstructuredGrid(cells, [CellType.POLYHEDRON], points)


def Pixel() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single pixel.

    This cell corresponds to the :attr:`pyvista.CellType.PIXEL` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single pixel.

    Examples
    --------
    Create and plot a single pixel.

    >>> from pyvista import examples
    >>> grid = examples.cells.Pixel()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([4, 0, 1, 2, 3])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0., 0., 0.],
                     [1., 0., 0.],
                     [0., 1., 0.],
                     [1., 1., 0.]])

    >>> grid.celltypes  # same as pyvista.CellType.PIXEL
    array([8], dtype=uint8)

    """
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.PIXEL], points)


def Quadrilateral() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadrilateral.

    This cell corresponds to the :attr:`pyvista.CellType.QUAD` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadrilateral.

    Examples
    --------
    Create and plot a single quad.

    >>> from pyvista import examples
    >>> grid = examples.cells.Quadrilateral()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([4, 0, 1, 2, 3])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0., 0., 0.],
                     [1., 0., 0.],
                     [1., 1., 0.],
                     [0., 1., 0.]])

    >>> grid.celltypes  # same as pyvista.CellType.QUAD
    array([9], dtype=uint8)

    """
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.QUAD], points)


def Tetrahedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Tetrahedron.

    This cell corresponds to the :attr:`pyvista.CellType.TETRA` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Tetrahedron.

    Examples
    --------
    Create and plot a single tetrahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.Tetrahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([4, 0, 1, 2, 3])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[ 0.35355339, -0.35355339, -0.35355339],
                     [ 0.35355339,  0.35355339,  0.35355339],
                     [-0.35355339,  0.35355339, -0.35355339],
                     [-0.35355339, -0.35355339,  0.35355339]])

    >>> grid.celltypes  # same as pyvista.CellType.TETRA
    array([10], dtype=uint8)

    """
    # Original points
    points = np.array(
        [
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )

    # Normalize points to have edge length one
    edge_length = np.linalg.norm(points[1] - points[0])
    points /= edge_length

    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.TETRA], points)


def Hexahedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single hexahedron.

    This cell corresponds to the :attr:`pyvista.CellType.HEXAHEDRON` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single hexahedron.

    Examples
    --------
    Create and plot a single hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.Hexahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([8, 0, 1, 2, 3, 4, 5, 6, 7])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0., 0., 0.],
                     [1., 0., 0.],
                     [1., 1., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.],
                     [1., 0., 1.],
                     [1., 1., 1.],
                     [0., 1., 1.]])

    >>> grid.celltypes  # same as pyvista.CellType.HEXAHEDRON
    array([12], dtype=uint8)

    """
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.HEXAHEDRON], points)


def HexagonalPrism() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single hexagonal prism.

    This cell corresponds to the :attr:`pyvista.CellType.HEXAGONAL_PRISM` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single hexagonal prism.

    Examples
    --------
    Create and plot a single hexagonal prism.

    >>> from pyvista import examples
    >>> grid = examples.cells.HexagonalPrism()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([12,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[ 0. ,  0. ,  1. ],
                     [-0.5,  0.5,  1. ],
                     [ 0. ,  1. ,  1. ],
                     [ 1. ,  1. ,  1. ],
                     [ 1.5,  0.5,  1. ],
                     [ 1. ,  0. ,  1. ],
                     [ 0. ,  0. ,  0. ],
                     [-0.5,  0.5,  0. ],
                     [ 0. ,  1. ,  0. ],
                     [ 1. ,  1. ,  0. ],
                     [ 1.5,  0.5,  0. ],
                     [ 1. ,  0. ,  0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.HEXAGONAL_PRISM
    array([16], dtype=uint8)

    """
    points = [
        # Top face (z=1.0)
        [0.0, 0.0, 1.0],
        [-0.5, 0.5, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.5, 0.5, 1.0],
        [1.0, 0.0, 1.0],
        # Bottom face (z=0.0)
        [0.0, 0.0, 0.0],
        [-0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.0, 0.0, 0.0],
    ]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.HEXAGONAL_PRISM], points)


def Wedge() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single wedge.

    This cell corresponds to the :attr:`pyvista.CellType.WEDGE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single wedge.

    Examples
    --------
    Create and plot a single wedge.

    >>> from pyvista import examples
    >>> grid = examples.cells.Wedge()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([6, 0, 1, 2, 3, 4, 5])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.       , 1.       , 0.       ],
                     [0.       , 0.       , 0.       ],
                     [0.       , 0.5      , 0.8660254],
                     [1.       , 1.       , 0.       ],
                     [1.       , 0.       , 0.       ],
                     [1.       , 0.5      , 0.8660254]])

    >>> grid.celltypes  # same as pyvista.CellType.WEDGE
    array([13], dtype=uint8)

    """
    R32 = np.sqrt(3) / 2
    points = [[0, 1, 0], [0, 0, 0], [0, 0.5, R32], [1, 1, 0], [1, 0.0, 0.0], [1, 0.5, R32]]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.WEDGE], points)


def PentagonalPrism() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single pentagonal prism.

    This cell corresponds to the :attr:`pyvista.CellType.PENTAGONAL_PRISM` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single pentagonal prism.

    Examples
    --------
    Create and plot a single pentagonal prism.

    >>> from pyvista import examples
    >>> grid = examples.cells.PentagonalPrism()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[1., 0., 0.],
                     [3., 0., 0.],
                     [4., 2., 0.],
                     [2., 4., 0.],
                     [0., 2., 0.],
                     [1., 0., 4.],
                     [3., 0., 4.],
                     [4., 2., 4.],
                     [2., 4., 4.],
                     [0., 2., 4.]])

    >>> grid.celltypes  # same as pyvista.CellType.PENTAGONAL_PRISM
    array([15], dtype=uint8)

    """
    points = [
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 2.0, 0.0],
        [2.0, 4.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 4.0],
        [3.0, 0.0, 4.0],
        [4.0, 2.0, 4.0],
        [2.0, 4.0, 4.0],
        [0.0, 2.0, 4.0],
    ]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.PENTAGONAL_PRISM], points)


def Pyramid() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single pyramid.

    This cell corresponds to the :attr:`pyvista.CellType.PYRAMID` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single pyramid.

    Examples
    --------
    Create and plot a single pyramid.

    >>> from pyvista import examples
    >>> grid = examples.cells.Pyramid()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([5, 0, 1, 2, 3, 4])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[ 0.5       ,  0.5       ,  0.        ],
                     [-0.5       ,  0.5       ,  0.        ],
                     [-0.5       , -0.5       ,  0.        ],
                     [ 0.5       , -0.5       ,  0.        ],
                     [ 0.        ,  0.        ,  0.70710678]])

    >>> grid.celltypes  # same as pyvista.CellType.PYRAMID
    array([14], dtype=uint8)

    """
    points = [
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.0, 0.0, np.sqrt(2) / 2],
    ]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.PYRAMID], points)


def Voxel() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single voxel.

    This cell corresponds to the :attr:`pyvista.CellType.VOXEL` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single voxel.

    Examples
    --------
    Create and plot a single voxel.

    >>> from pyvista import examples
    >>> grid = examples.cells.Voxel()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([8, 0, 1, 2, 3, 4, 5, 6, 7])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0., 0., 0.],
                     [1., 0., 0.],
                     [1., 1., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.],
                     [1., 0., 1.],
                     [1., 1., 1.],
                     [0., 1., 1.]])

    >>> grid.celltypes  # same as pyvista.CellType.VOXEL
    array([11], dtype=uint8)

    """
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.VOXEL], points)


def QuadraticEdge() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic edge.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_EDGE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic edge.

    Examples
    --------
    Create and plot a single quadratic edge.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticEdge()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([3, 0, 1, 2])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [0.5, 0. , 0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_EDGE
    array([21], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticEdge())


def QuadraticTriangle() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic triangle.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_TRIANGLE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic triangle.

    Examples
    --------
    Create and plot a single quadratic triangle.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticTriangle()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([6, 0, 1, 2, 3, 4, 5])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [0. , 1. , 0. ],
                     [0.5, 0. , 0. ],
                     [0.5, 0.5, 0. ],
                     [0. , 0.5, 0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_TRIANGLE
    array([22], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticTriangle())


def QuadraticQuadrilateral() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic quadrilateral.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_QUAD` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic quadrilateral.

    Examples
    --------
    Create and plot a single quadratic quadrilateral.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticQuadrilateral()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([8, 0, 1, 2, 3, 4, 5, 6, 7])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [1. , 1. , 0. ],
                     [0. , 1. , 0. ],
                     [0.5, 0. , 0. ],
                     [1. , 0.5, 0. ],
                     [0.5, 1. , 0. ],
                     [0. , 0.5, 0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_QUAD
    array([23], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticQuad())


def QuadraticPolygon() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic polygon.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_POLYGON` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic polygon.

    Examples
    --------
    Create and plot a single quadratic polygon.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticPolygon()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([8, 0, 1, 2, 3, 4, 5, 6, 7])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0., 0., 0.],
                     [2., 0., 0.],
                     [2., 2., 0.],
                     [0., 2., 0.],
                     [1., 0., 0.],
                     [3., 1., 0.],
                     [1., 2., 0.],
                     [0., 1., 0.]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_POLYGON
    array([36], dtype=uint8)

    """
    points = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.QUADRATIC_POLYGON], points)


def QuadraticTetrahedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic tetrahedron.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_TETRA` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic tetrahedron.

    Examples
    --------
    Create and plot a single quadratic tetrahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticTetrahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [0.5, 0. , 0. ],
                     [0.5, 0.5, 0. ],
                     [0. , 0.5, 0. ],
                     [0. , 0. , 0.5],
                     [0.5, 0. , 0.5],
                     [0. , 0.5, 0.5]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_TETRA
    array([24], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticTetra())


def QuadraticHexahedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic hexahedron.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_HEXAHEDRON` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic hexahedron.

    Examples
    --------
    Create and plot a single quadratic hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticHexahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([20,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [1. , 1. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [1. , 0. , 1. ],
                     [1. , 1. , 1. ],
                     [0. , 1. , 1. ],
                     [0.5, 0. , 0. ],
                     [1. , 0.5, 0. ],
                     [0.5, 1. , 0. ],
                     [0. , 0.5, 0. ],
                     [0.5, 0. , 1. ],
                     [1. , 0.5, 1. ],
                     [0.5, 1. , 1. ],
                     [0. , 0.5, 1. ],
                     [0. , 0. , 0.5],
                     [1. , 0. , 0.5],
                     [1. , 1. , 0.5],
                     [0. , 1. , 0.5]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_HEXAHEDRON
    array([25], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticHexahedron())


def QuadraticWedge() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic wedge.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_WEDGE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic wedge.

    Examples
    --------
    Create and plot a single quadratic wedge.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticWedge()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [1. , 0. , 1. ],
                     [0. , 1. , 1. ],
                     [0.5, 0. , 0. ],
                     [0.5, 0.5, 0. ],
                     [0. , 0.5, 0. ],
                     [0.5, 0. , 1. ],
                     [0.5, 0.5, 1. ],
                     [0. , 0.5, 1. ],
                     [0. , 0. , 0.5],
                     [1. , 0. , 0.5],
                     [0. , 1. , 0.5]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_WEDGE
    array([26], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticWedge())


def QuadraticPyramid() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic pyramid.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_PYRAMID` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic pyramid.

    Examples
    --------
    Create and plot a single quadratic pyramid.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticPyramid()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [1. , 1. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [0.5, 0. , 0. ],
                     [1. , 0.5, 0. ],
                     [0.5, 1. , 0. ],
                     [0. , 0.5, 0. ],
                     [0. , 0. , 0.5],
                     [1. , 0. , 0.5],
                     [1. , 1. , 0.5],
                     [0. , 1. , 0.5]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_POLYGON
    array([27], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticPyramid())


def BiQuadraticQuadrilateral() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single biquadratic quadrilateral.

    This cell corresponds to the :attr:`pyvista.CellType.BIQUADRATIC_QUAD` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single biquadratic quadrilateral.

    Examples
    --------
    Create and plot a single biquadratic quadrilateral.

    >>> from pyvista import examples
    >>> grid = examples.cells.BiQuadraticQuadrilateral()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([9, 0, 1, 2, 3, 4, 5, 6, 7, 8])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [1. , 1. , 0. ],
                     [0. , 1. , 0. ],
                     [0.5, 0. , 0. ],
                     [1. , 0.5, 0. ],
                     [0.5, 1. , 0. ],
                     [0. , 0.5, 0. ],
                     [0.5, 0.5, 0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.BIQUADRATIC_QUAD
    array([28], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkBiQuadraticQuad())


def TriQuadraticHexahedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single triquadratic hexahedron.

    This cell corresponds to the :attr:`pyvista.CellType.TRIQUADRATIC_HEXAHEDRON` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single triquadratic hexahedron.

    Examples
    --------
    Create and plot a single triquadratic hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.TriQuadraticHexahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([27,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [1. , 1. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [1. , 0. , 1. ],
                     [1. , 1. , 1. ],
                     [0. , 1. , 1. ],
                     [0.5, 0. , 0. ],
                     [1. , 0.5, 0. ],
                     [0.5, 1. , 0. ],
                     [0. , 0.5, 0. ],
                     [0.5, 0. , 1. ],
                     [1. , 0.5, 1. ],
                     [0.5, 1. , 1. ],
                     [0. , 0.5, 1. ],
                     [0. , 0. , 0.5],
                     [1. , 0. , 0.5],
                     [1. , 1. , 0.5],
                     [0. , 1. , 0.5],
                     [0. , 0.5, 0.5],
                     [1. , 0.5, 0.5],
                     [0.5, 0. , 0.5],
                     [0.5, 1. , 0.5],
                     [0.5, 0.5, 0. ],
                     [0.5, 0.5, 1. ],
                     [0.5, 0.5, 0.5]])

    >>> grid.celltypes  # same as pyvista.CellType.TRIQUADRATIC_HEXAHEDRON
    array([29], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkTriQuadraticHexahedron())


def TriQuadraticPyramid() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single triquadratic pyramid.

    This cell corresponds to the :attr:`pyvista.CellType.TRIQUADRATIC_PYRAMID` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single triquadratic pyramid.

    Examples
    --------
    Create and plot a single triquadratic pyramid.

    >>> from pyvista import examples
    >>> grid = examples.cells.TriQuadraticPyramid()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.5       ],
                     [1.        , 0.        , 0.5       ],
                     [1.        , 1.        , 0.5       ],
                     [0.        , 1.        , 0.5       ],
                     [0.5       , 0.5       , 1.        ],
                     [0.5       , 0.        , 0.5       ],
                     [1.        , 0.5       , 0.5       ],
                     [0.5       , 1.        , 0.5       ],
                     [0.        , 0.5       , 0.5       ],
                     [0.25      , 0.25      , 0.75      ],
                     [0.75      , 0.25      , 0.75      ],
                     [0.75      , 0.75      , 0.75      ],
                     [0.25      , 0.75      , 0.75      ],
                     [0.5       , 0.5       , 0.5       ],
                     [0.5       , 0.16666667, 0.66666667],
                     [0.83333333, 0.5       , 0.66666667],
                     [0.5       , 0.83333333, 0.66666667],
                     [0.16666667, 0.5       , 0.66666667],
                     [0.5       , 0.5       , 0.625     ]])

    >>> grid.celltypes  # same as pyvista.CellType.TRIQUADRATIC_PYRAMID
    array([37], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkTriQuadraticPyramid())


def QuadraticLinearQuadrilateral() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic linear quadrilateral.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_LINEAR_QUAD` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic linear quadrilateral.

    Examples
    --------
    Create and plot a single quadratic linear quadrilateral.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticLinearQuadrilateral()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([6, 0, 1, 2, 3, 4, 5])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [1. , 1. , 0. ],
                     [0. , 1. , 0. ],
                     [0.5, 0. , 0. ],
                     [0.5, 1. , 0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_LINEAR_QUAD
    array([30], dtype=uint8)

    """  # noqa: E501
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticLinearQuad())


def QuadraticLinearWedge() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single quadratic linear wedge.

    This cell corresponds to the :attr:`pyvista.CellType.QUADRATIC_LINEAR_WEDGE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single quadratic linear wedge.

    Examples
    --------
    Create and plot a single quadratic linear wedge.

    >>> from pyvista import examples
    >>> grid = examples.cells.QuadraticLinearWedge()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([12,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [1. , 0. , 1. ],
                     [0. , 1. , 1. ],
                     [0.5, 0. , 0. ],
                     [0.5, 0.5, 0. ],
                     [0. , 0.5, 0. ],
                     [0.5, 0. , 1. ],
                     [0.5, 0.5, 1. ],
                     [0. , 0.5, 1. ]])

    >>> grid.celltypes  # same as pyvista.CellType.QUADRATIC_LINEAR_WEDGE
    array([31], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkQuadraticLinearWedge())


def BiQuadraticQuadraticWedge() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single biquadratic quadratic wedge.

    This cell corresponds to the :attr:`pyvista.CellType.BIQUADRATIC_QUADRATIC_WEDGE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single biquadratic quadratic wedge.

    Examples
    --------
    Create and plot a single biquadratic quadratic wedge.

    >>> from pyvista import examples
    >>> grid = examples.cells.BiQuadraticQuadraticWedge()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([18,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [1. , 0. , 1. ],
                     [0. , 1. , 1. ],
                     [0.5, 0. , 0. ],
                     [0.5, 0.5, 0. ],
                     [0. , 0.5, 0. ],
                     [0.5, 0. , 1. ],
                     [0.5, 0.5, 1. ],
                     [0. , 0.5, 1. ],
                     [0. , 0. , 0.5],
                     [1. , 0. , 0.5],
                     [0. , 1. , 0.5],
                     [0.5, 0. , 0.5],
                     [0.5, 0.5, 0.5],
                     [0. , 0.5, 0.5]])

    >>> grid.celltypes  # same as pyvista.CellType.BIQUADRATIC_QUADRATIC_WEDGE
    array([32], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkBiQuadraticQuadraticWedge())


def BiQuadraticQuadraticHexahedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single biquadratic quadratic hexahedron.

    This cell corresponds to the :attr:`pyvista.CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON`
    cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single biquadratic quadratic hexahedron.

    Examples
    --------
    Create and plot a single biquadratic quadratic hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.BiQuadraticQuadraticHexahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([24,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21, 22, 23])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0. , 0. , 0. ],
                     [1. , 0. , 0. ],
                     [1. , 1. , 0. ],
                     [0. , 1. , 0. ],
                     [0. , 0. , 1. ],
                     [1. , 0. , 1. ],
                     [1. , 1. , 1. ],
                     [0. , 1. , 1. ],
                     [0.5, 0. , 0. ],
                     [1. , 0.5, 0. ],
                     [0.5, 1. , 0. ],
                     [0. , 0.5, 0. ],
                     [0.5, 0. , 1. ],
                     [1. , 0.5, 1. ],
                     [0.5, 1. , 1. ],
                     [0. , 0.5, 1. ],
                     [0. , 0. , 0.5],
                     [1. , 0. , 0.5],
                     [1. , 1. , 0.5],
                     [0. , 1. , 0.5],
                     [0. , 0.5, 0.5],
                     [1. , 0.5, 0.5],
                     [0.5, 0. , 0.5],
                     [0.5, 1. , 0.5]])

    >>> grid.celltypes  # same as pyvista.CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON
    array([33], dtype=uint8)

    """  # noqa: E501
    return _make_isoparametric_unstructured_grid(_vtk.vtkBiQuadraticQuadraticHexahedron())


def BiQuadraticTriangle() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single biquadratic triangle.

    This cell corresponds to the :attr:`pyvista.CellType.BIQUADRATIC_TRIANGLE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single biquadratic triangle.

    Examples
    --------
    Create and plot a single biquadratic triangle.

    >>> from pyvista import examples
    >>> grid = examples.cells.BiQuadraticTriangle()
    >>> examples.plot_cell(grid, cpos='xy')

    List the grid's cells.

    >>> grid.cells
    array([7, 0, 1, 2, 3, 4, 5, 6])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.5       , 0.        , 0.        ],
                     [0.5       , 0.5       , 0.        ],
                     [0.        , 0.5       , 0.        ],
                     [0.33333333, 0.33333333, 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.BIQUADRATIC_TRIANGLE
    array([34], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkBiQuadraticTriangle())


def CubicLine() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single cubic line.

    This cell corresponds to the :attr:`pyvista.CellType.CUBIC_LINE` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single cubic line.

    Examples
    --------
    Create and plot a single cubic line.

    >>> from pyvista import examples
    >>> grid = examples.cells.CubicLine()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([4, 0, 1, 2, 3])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[-1.        ,  0.        ,  0.        ],
                     [ 1.        ,  0.        ,  0.        ],
                     [-0.33333333,  0.        ,  0.        ],
                     [ 0.33333333,  0.        ,  0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.CUBIC_LINE
    array([35], dtype=uint8)

    """
    return _make_isoparametric_unstructured_grid(_vtk.vtkCubicLine())


def _make_isoparametric_unstructured_grid(vtk_cell: _vtk.vtkCell):
    cell = pv.Cell(vtk_cell)  # type: ignore[abstract]

    # Create points
    pcoords = cell.GetParametricCoords()
    points = np.zeros((cell.n_points, 3))
    for i in range(len(points)):
        points[i] = (pcoords[3 * i], pcoords[3 * i + 1], pcoords[3 * i + 2])

    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [cell.type], points)
