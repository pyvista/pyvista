"""Contains a variety of cells to serve as examples."""
from pyvista import CellType, UnstructuredGrid


def Tetrahedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Tetrahedron.

    This cell corresponds to the :attr:`pyvista.CellType.TETRA` cell type.

    Returns
    -------
    UnstructuredGrid
        UnstructuredGrid containing a single Tetrahedron.

    Examples
    --------
    Create and plot a single tetrahedron.

    >>> import pyvista as pv
    >>> grid = pv.cells.Tetrahedron()
    >>> grid.plot(show_edges=True)

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
    cells = [len(points)] + list(range(len(points)))
    return UnstructuredGrid(cells, [CellType.TETRA], points)


def Hexahedron() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single hexahedron.

    This cell corresponds to the :attr:`pyvista.CellType.HEXAHEDRON` cell type.

    Returns
    -------
    UnstructuredGrid
        UnstructuredGrid containing a single hexahedron.

    Examples
    --------
    Create and plot a single hexahedron.

    >>> import pyvista as pv
    >>> grid = pv.cells.Hexahedron()
    >>> grid.plot(show_edges=True)

    List the grid's cells.

    >>> grid.cells
    array([12,  0,  1,  2,  3,  4,  5,  6,  7])

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
    cells = [len(points)] + list(range(len(points)))
    return UnstructuredGrid(cells, [CellType.HEXAHEDRON], points)


def HexagonalPrism() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single hexagonal prism.

    This cell corresponds to the :attr:`pyvista.CellType.HEXAGONAL_PRISM` cell type.

    Returns
    -------
    UnstructuredGrid
        UnstructuredGrid containing a single hexagonal prism.

    Examples
    --------
    Create and plot a single hexagonal prism.

    >>> import pyvista as pv
    >>> grid = pv.cells.HexagonalPrism()
    >>> grid.plot(show_edges=True)

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
    cells = [len(points)] + list(range(len(points)))
    return UnstructuredGrid(cells, [CellType.HEXAGONAL_PRISM], points)


def Wedge() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single wedge.

    This cell corresponds to the :attr:`pyvista.CellType.WEDGE` cell type.

    Returns
    -------
    UnstructuredGrid
        UnstructuredGrid containing a single wedge.

    Examples
    --------
    Create and plot a single wedge.

    >>> import pyvista as pv
    >>> grid = pv.cells.Wedge()
    >>> grid.plot(show_edges=True)

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
    cells = [len(points)] + list(range(len(points)))
    return UnstructuredGrid(cells, [CellType.WEDGE], points)


def PentagonalPrism() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single pentagonal prism.

    This cell corresponds to the :attr:`pyvista.CellType.PENTAGONAL_PRISM` cell type.

    Returns
    -------
    UnstructuredGrid
        UnstructuredGrid containing a single pentagonal prism.

    Examples
    --------
    Create and plot a single pentagonal prism.

    >>> import pyvista as pv
    >>> grid = pv.cells.PentagonalPrism()
    >>> grid.plot(show_edges=True)

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
    cells = [len(points)] + list(range(len(points)))
    return UnstructuredGrid(cells, [CellType.PENTAGONAL_PRISM], points)
