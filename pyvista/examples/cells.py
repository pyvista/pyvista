"""Contains a variety of cells to serve as examples."""

from __future__ import annotations

import pyvista
from pyvista import CellType
from pyvista import UnstructuredGrid


def plot_cell(grid, cpos=None, **kwargs):
    """Plot a :class:`pyvista.UnstructuredGrid` while displaying cell indices.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Unstructured grid (ideally) containing a single cell.

    cpos : str, optional
        Camera position.

    **kwargs : dict, optional
        Additional keyword arguments when showing. See :func:`pyvista.Plotter.show`.

    Examples
    --------
    Create and plot a single hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.Hexahedron()
    >>> examples.plot_cell(grid)

    """
    pl = pyvista.Plotter()
    pl.add_mesh(grid, opacity=0.5)
    edges = grid.extract_all_edges()
    if edges.n_cells:
        pl.add_mesh(grid.extract_all_edges(), line_width=5, color='k', render_lines_as_tubes=True)
    pl.add_points(grid, render_points_as_spheres=True, point_size=30, color='r')
    pl.add_point_labels(
        grid.points,
        range(grid.n_points),
        always_visible=True,
        fill_shape=False,
        margin=0,
        shape_opacity=0.0,
        font_size=20,
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
    pyvista_ndarray([[1., 0., 0.],
                     [0., 0., 0.],
                     [0., 1., 0.]])

    >>> grid.celltypes  # same as pyvista.CellType.TRIANGLE
    array([5], dtype=uint8)

    """
    points = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
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
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 3.0, 0.0],
        [0.0, 3.0, 0.0],
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
    cells = [4, 3, 0, 1, 2, 3, 0, 1, 3, 3, 0, 2, 3, 3, 1, 2, 3]
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
    pyvista_ndarray([[ 1.,  1.,  1.],
                     [ 1., -1., -1.],
                     [-1.,  1., -1.],
                     [-1., -1.,  1.]])

    >>> grid.celltypes  # same as pyvista.CellType.TETRA
    array([10], dtype=uint8)

    """
    points = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
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
                     [ 1. ,  0. ,  1. ],
                     [ 1.5,  0.5,  1. ],
                     [ 1. ,  1. ,  1. ],
                     [ 0. ,  1. ,  1. ],
                     [-0.5,  0.5,  1. ],
                     [ 0. ,  0. ,  0. ],
                     [ 1. ,  0. ,  0. ],
                     [ 1.5,  0.5,  0. ],
                     [ 1. ,  1. ,  0. ],
                     [ 0. ,  1. ,  0. ],
                     [-0.5,  0.5,  0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.HEXAGONAL_PRISM
    array([16], dtype=uint8)

    """
    points = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.5, 0.5, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [-0.5, 0.5, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-0.5, 0.5, 0.0],
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
    pyvista_ndarray([[0. , 1. , 0. ],
                     [0. , 0. , 0. ],
                     [0. , 0.5, 0.5],
                     [1. , 1. , 0. ],
                     [1. , 0. , 0. ],
                     [1. , 0.5, 0.5]])

    >>> grid.celltypes  # same as pyvista.CellType.WEDGE
    array([13], dtype=uint8)

    """
    points = [[0, 1, 0], [0, 0, 0], [0, 0.5, 0.5], [1, 1, 0], [1, 0.0, 0.0], [1, 0.5, 0.5]]
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
    pyvista_ndarray([[ 1.        ,  1.        ,  0.        ],
                     [-1.        ,  1.        ,  0.        ],
                     [-1.        , -1.        ,  0.        ],
                     [ 1.        , -1.        ,  0.        ],
                     [ 0.        ,  0.        ,  1.60803807]])

    >>> grid.celltypes  # same as pyvista.CellType.PYRAMID
    array([14], dtype=uint8)

    """
    points = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 0.0, 1.60803807],
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
