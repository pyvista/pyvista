"""Contains a variety of cells to serve as examples.

Functions that create single cell :class:`pyvista.UnstructuredGrid` objects which can
be used to learn about VTK :class:`cell types <pyvista.CellType>`.

"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
import itertools
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import numpy as np

import pyvista as pv
from pyvista import CellType
from pyvista import UnstructuredGrid
from pyvista._warn_external import warn_external
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk

if TYPE_CHECKING:
    from pyvista import DataSet
    from pyvista import MultiBlock
    from pyvista import VectorLike
    from pyvista.plotting._typing import CameraPositionOptions


_NOT_SUPPORTED_CELL_SOURCE = [
    CellType.EMPTY_CELL,
    CellType.VERTEX,
    CellType.POLY_VERTEX,
    CellType.POLY_LINE,
    CellType.TRIANGLE_STRIP,
    CellType.QUADRATIC_LINEAR_QUAD,
    CellType.QUADRATIC_LINEAR_WEDGE,
    CellType.BIQUADRATIC_QUADRATIC_WEDGE,
    CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON,
    CellType.BIQUADRATIC_TRIANGLE,
    CellType.QUADRATIC_POLYGON,
    CellType.CONVEX_POINT_SET,
]

_NOT_SUPPORTED_PARAMETRIC = [
    CellType.POLY_VERTEX,
    CellType.POLY_LINE,
    CellType.TRIANGLE_STRIP,
    CellType.POLYGON,
    CellType.QUADRATIC_POLYGON,
    CellType.CONVEX_POINT_SET,
    CellType.POLYHEDRON,
]


def plot_cell(
    grid: DataSet | MultiBlock,
    cpos: CameraPositionOptions | None = None,
    *,
    line_width: int | None = None,
    point_size: int | None = None,
    font_size: int | None = None,
    show_normals: bool = False,
    normals_scale: float | None = None,
    **kwargs,
):
    """Plot a mesh while displaying cell indices.

    .. versionchanged:: 0.45
        The default line width, point size, and font size are increased from ``5``, ``30``
        and ``20`` to ``10``, ``80``, and ``50``, respectively.

    .. versionchanged:: 0.47
        The default line width, point size, and font size are restored to their original
        values prior to version 0.45. These values can now be customized with keywords.

    Parameters
    ----------
    grid : DataSet | MultiBlock
        Dataset containing one single cell (ideally), though plotting a mesh with multiple cells
        is supported.

        .. versionchanged:: 0.47
            Plotting :class:`~pyvista.PolyData` is now supported.

        .. versionchanged:: 0.48
            Plotting :class:`~pyvista.MultiBlock` is now supported.

    cpos : str, optional
        Camera position. Byx default, an ``'xy'`` view is used for 2D planar inputs; otherwise,
        it's set to ``azimuth=20`` and ``elevation=-20``.

        .. versionchanged:: 0.48

            An ``'xy'`` view is now used by default for planar inputs.

    line_width : int, default: 5
        Line width of the cell's edges.

        .. versionadded:: 0.47

    point_size : int, default: 30
        Size of the cell's points.

        .. versionadded:: 0.47

    font_size : int, default: 20
        Size of the point labels.

        .. versionadded:: 0.47

    show_normals : bool, optional
        Show the face normals of the cell. Only applies to 2D or 3D cells.
        Cell faces with correct orientation should have the normal pointing outward.

        The size of the normals is controlled by ``normals_scale``.

        .. versionadded:: 0.47

    normals_scale : float, default: 0.1
        Scale factor used when ``show_normals`` is enabled. The normals are
        scaled proportional to the diagonal length of the input ``grid``.

        .. versionadded:: 0.47

    **kwargs : dict, optional
        Additional keyword arguments when showing. See :func:`pyvista.Plotter.show`.

    Examples
    --------
    Create and plot a single hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.Hexahedron()
    >>> examples.plot_cell(grid)

    Show normals and customize the size of various elements in the rendering.

    >>> examples.plot_cell(
    ...     grid,
    ...     show_normals=True,
    ...     normals_scale=0.2,
    ...     line_width=8,
    ...     point_size=50,
    ...     font_size=30,
    ... )

    """
    config = pv.global_theme._plot_cell
    line_width_ = config._line_width if line_width is None else line_width
    point_size_ = config._point_size if point_size is None else point_size
    font_size_ = config._font_size if font_size is None else font_size
    normals_scale_ = config._normals_scale if normals_scale is None else normals_scale

    def _extract_surface(cell_):
        if cell_.type == pv.CellType.POLYHEDRON:
            # For Polyhedron, we don't use ``extract_surface`` directly because that may alter
            # the face orientation, so we iterate over each face directly to create separate
            # PolyData faces instead.
            faces = pv.MultiBlock()
            for i in range(cell_.n_faces):
                face = cell_.GetFace(i)
                face_n_points = face.GetNumberOfPoints()
                point_ids = [face.GetPointId(i) for i in range(face_n_points)]
                poly = pv.PolyData(grid.points, [face_n_points, *point_ids])
                faces.append(poly)
            return faces.extract_surface(algorithm=None, pass_pointid=False, pass_cellid=False)
        return cell_.cast_to_unstructured_grid().extract_surface(
            algorithm=None, pass_pointid=False, pass_cellid=False
        )

    if isinstance(grid, pv.MultiBlock):
        grid = grid.combine()
    elif not isinstance(grid, pv.UnstructuredGrid):
        grid = grid.cast_to_unstructured_grid()

    pl = pv.Plotter()
    has_2d_cells_only = True
    for cell in grid.cell:
        dimension = cell.dimension
        if dimension != 2:
            has_2d_cells_only = False

        # Use existing grid if it's already a grid with one cell
        cell_as_grid = grid if grid.n_cells == 1 else cell.cast_to_unstructured_grid()
        pl.add_mesh(cell_as_grid, opacity=0.5)
        if cell.type == CellType.CONVEX_POINT_SET:
            # Cell does not actually have edges, so convert it to a surface first
            edges = cell_as_grid.extract_surface(algorithm=None).extract_all_edges()
        else:
            edges = cell_as_grid.extract_all_edges()
        if edges.n_cells or dimension == 1:
            pl.add_mesh(
                cell_as_grid,
                style='wireframe',
                line_width=line_width_,
                color='k',
                render_lines_as_tubes=True,
            )
        pl.add_points(
            cell.points, render_points_as_spheres=True, point_size=point_size_, color='r'
        )
        pl.add_point_labels(
            cell.points,
            cell.point_ids,
            always_visible=True,
            fill_shape=False,
            margin=0,
            shape_opacity=0.0,
            font_size=font_size_,
        )

        if show_normals and dimension >= 2:
            surface = _extract_surface(cell)
            surface = surface.triangulate() if cell.type is CellType.TRIANGLE_STRIP else surface
            pl.add_arrows(
                surface.cell_centers().points,
                surface.cell_normals,
                mag=grid.length * normals_scale_,
                color='yellow',
                show_scalar_bar=False,
            )

    pl.enable_anti_aliasing()
    if cpos is None:
        # Use xy view if we have all 2D cells lying in the xy plane
        if has_2d_cells_only and grid.dimensionality == 2 and np.isclose(grid.bounds_size[2], 0.0):
            pl.view_xy()  # type: ignore[call-arg]
        else:
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


def ConvexPointSet() -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single convex point set.

    This represents a set of 3D vertices as a single convex cell.

    This cell corresponds to the :attr:`pyvista.CellType.CONVEX_POINT_SET` cell type.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single convex point set.

    Examples
    --------
    Create and plot a single convex point set.

    >>> from pyvista import examples
    >>> grid = examples.cells.ConvexPointSet()
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

    >>> grid.celltypes  # same as pyvista.CellType.CONVEX_POINT_SET
    array([41], dtype=uint8)

    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0.4], [0, 1, 0.6]]
    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [CellType.CONVEX_POINT_SET], points)


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
    >>> examples.plot_cell(grid)

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
    >>> examples.plot_cell(grid)

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
                     [ 1.4,  0.5,  0. ],
                     [ 1. ,  1. ,  0. ],
                     [ 0.6,  1.2,  0. ],
                     [ 0. ,  0.8,  0. ]])

    >>> grid.celltypes  # same as pyvista.CellType.POLYGON
    array([7], dtype=uint8)

    """
    points = [[0, 0, 0], [1, -0.1, 0], [1.4, 0.5, 0], [1, 1, 0], [0.6, 1.2, 0], [0, 0.8, 0]]
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
    >>> examples.plot_cell(grid)

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
    >>> examples.plot_cell(grid)

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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_EDGE))


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
    >>> examples.plot_cell(grid)

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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_TRIANGLE))


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
    >>> examples.plot_cell(grid)

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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_QUAD))


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
    >>> examples.plot_cell(grid)

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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_TETRA))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_HEXAHEDRON))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_WEDGE))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_PYRAMID))


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
    >>> examples.plot_cell(grid)

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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.BIQUADRATIC_QUAD))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.TRIQUADRATIC_HEXAHEDRON))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.TRIQUADRATIC_PYRAMID))


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
    >>> examples.plot_cell(grid)

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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_LINEAR_QUAD))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.QUADRATIC_LINEAR_WEDGE))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.BIQUADRATIC_QUADRATIC_WEDGE))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON))


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
    >>> examples.plot_cell(grid)

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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.BIQUADRATIC_TRIANGLE))


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
    return cast('UnstructuredGrid', _isoparametric_grid(CellType.CUBIC_LINE))


def LagrangeCurve(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Lagrange curve.

    This cell corresponds to the :attr:`pyvista.CellType.LAGRANGE_CURVE` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Lagrange curve.

    Examples
    --------
    Create and plot a single Lagrange curve.

    >>> from pyvista import examples
    >>> grid = examples.cells.LagrangeCurve()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([4, 0, 1, 2, 3])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.LAGRANGE_CURVE
    array([68], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid', _vtkCellTypeSource(CellType.LAGRANGE_CURVE, cell_order=cell_order)
    )


def LagrangeTriangle(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Lagrange triangle.

    This cell corresponds to the :attr:`pyvista.CellType.LAGRANGE_TRIANGLE` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Lagrange triangle.

    Examples
    --------
    Create and plot a single Lagrange triangle.

    >>> from pyvista import examples
    >>> grid = examples.cells.LagrangeTriangle()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [0.66666669, 0.33333334, 0.        ],
                     [0.33333334, 0.66666669, 0.        ],
                     [0.        , 0.66666669, 0.        ],
                     [0.        , 0.33333334, 0.        ],
                     [0.33333334, 0.33333334, 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.LAGRANGE_TRIANGLE
    array([69], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid',
        _vtkCellTypeSource(CellType.LAGRANGE_TRIANGLE, cell_order=cell_order),
    )


def LagrangeQuadrilateral(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Lagrange quadrilateral.

    This cell corresponds to the :attr:`pyvista.CellType.LAGRANGE_QUADRILATERAL` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Lagrange quadrilateral.

    Examples
    --------
    Create and plot a single Lagrange quadrilateral.

    >>> from pyvista import examples
    >>> grid = examples.cells.LagrangeQuadrilateral()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([16,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [1.        , 1.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [1.        , 0.33333334, 0.        ],
                     [1.        , 0.66666669, 0.        ],
                     [0.33333334, 1.        , 0.        ],
                     [0.66666669, 1.        , 0.        ],
                     [0.        , 0.33333334, 0.        ],
                     [0.        , 0.66666669, 0.        ],
                     [0.33333334, 0.33333334, 0.        ],
                     [0.66666669, 0.33333334, 0.        ],
                     [0.33333334, 0.66666669, 0.        ],
                     [0.66666669, 0.66666669, 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.LAGRANGE_QUADRILATERAL
    array([70], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid',
        _vtkCellTypeSource(CellType.LAGRANGE_QUADRILATERAL, cell_order=cell_order),
    )


def LagrangeTetrahedron(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Lagrange tetrahedron.

    This cell corresponds to the :attr:`pyvista.CellType.LAGRANGE_TETRAHEDRON` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Lagrange tetrahedron.

    Examples
    --------
    Create and plot a single Lagrange tetrahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.LagrangeTetrahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([20,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [1.        , 1.        , 0.        ],
                     [0.5       , 0.5       , 0.5       ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [1.        , 0.33333334, 0.        ],
                     [1.        , 0.66666669, 0.        ],
                     [0.66666669, 0.66666669, 0.        ],
                     [0.33333334, 0.33333334, 0.        ],
                     [0.16666667, 0.16666667, 0.16666667],
                     [0.33333334, 0.33333334, 0.33333334],
                     [0.83333331, 0.16666667, 0.16666667],
                     [0.66666669, 0.33333334, 0.33333334],
                     [0.83333331, 0.83333331, 0.16666667],
                     [0.66666669, 0.66666669, 0.33333334],
                     [0.5       , 0.16666667, 0.16666667],
                     [0.83333331, 0.5       , 0.16666667],
                     [0.5       , 0.5       , 0.16666667],
                     [0.66666669, 0.33333334, 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.LAGRANGE_TETRAHEDRON
    array([71], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid',
        _vtkCellTypeSource(CellType.LAGRANGE_TETRAHEDRON, cell_order=cell_order),
    )


def LagrangeHexahedron(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Lagrange hexahedron.

    This cell corresponds to the :attr:`pyvista.CellType.LAGRANGE_HEXAHEDRON` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Lagrange hexahedron.

    Examples
    --------
    Create and plot a single Lagrange hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.LagrangeHexahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
           33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
           50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [1.        , 1.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.        , 0.        , 1.        ],
                     [1.        , 0.        , 1.        ],
                     [1.        , 1.        , 1.        ],
                     [0.        , 1.        , 1.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [1.        , 0.33333334, 0.        ],
                     [1.        , 0.66666669, 0.        ],
                     [0.33333334, 1.        , 0.        ],
                     [0.66666669, 1.        , 0.        ],
                     [0.        , 0.33333334, 0.        ],
                     [0.        , 0.66666669, 0.        ],
                     [0.33333334, 0.        , 1.        ],
                     [0.66666669, 0.        , 1.        ],
                     [1.        , 0.33333334, 1.        ],
                     [1.        , 0.66666669, 1.        ],
                     [0.33333334, 1.        , 1.        ],
                     [0.66666669, 1.        , 1.        ],
                     [0.        , 0.33333334, 1.        ],
                     [0.        , 0.66666669, 1.        ],
                     [0.        , 0.        , 0.33333334],
                     [0.        , 0.        , 0.66666669],
                     [1.        , 0.        , 0.33333334],
                     [1.        , 0.        , 0.66666669],
                     [1.        , 1.        , 0.33333334],
                     [1.        , 1.        , 0.66666669],
                     [0.        , 1.        , 0.33333334],
                     [0.        , 1.        , 0.66666669],
                     [0.        , 0.33333334, 0.33333334],
                     [0.        , 0.66666669, 0.33333334],
                     [0.        , 0.33333334, 0.66666669],
                     [0.        , 0.66666669, 0.66666669],
                     [1.        , 0.33333334, 0.33333334],
                     [1.        , 0.66666669, 0.33333334],
                     [1.        , 0.33333334, 0.66666669],
                     [1.        , 0.66666669, 0.66666669],
                     [0.33333334, 0.        , 0.33333334],
                     [0.66666669, 0.        , 0.33333334],
                     [0.33333334, 0.        , 0.66666669],
                     [0.66666669, 0.        , 0.66666669],
                     [0.33333334, 1.        , 0.33333334],
                     [0.66666669, 1.        , 0.33333334],
                     [0.33333334, 1.        , 0.66666669],
                     [0.66666669, 1.        , 0.66666669],
                     [0.33333334, 0.33333334, 0.        ],
                     [0.66666669, 0.33333334, 0.        ],
                     [0.33333334, 0.66666669, 0.        ],
                     [0.66666669, 0.66666669, 0.        ],
                     [0.33333334, 0.33333334, 1.        ],
                     [0.66666669, 0.33333334, 1.        ],
                     [0.33333334, 0.66666669, 1.        ],
                     [0.66666669, 0.66666669, 1.        ],
                     [0.33333334, 0.33333334, 0.33333334],
                     [0.66666669, 0.33333334, 0.33333334],
                     [0.33333334, 0.66666669, 0.33333334],
                     [0.66666669, 0.66666669, 0.33333334],
                     [0.33333334, 0.33333334, 0.66666669],
                     [0.66666669, 0.33333334, 0.66666669],
                     [0.33333334, 0.66666669, 0.66666669],
                     [0.66666669, 0.66666669, 0.66666669]])

    >>> grid.celltypes  # same as pyvista.CellType.LAGRANGE_HEXAHEDRON
    array([72], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid',
        _vtkCellTypeSource(CellType.LAGRANGE_HEXAHEDRON, cell_order=cell_order),
    )


def LagrangeWedge(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Lagrange wedge.

    This cell corresponds to the :attr:`pyvista.CellType.LAGRANGE_WEDGE` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Lagrange wedge.

    Examples
    --------
    Create and plot a single Lagrange wedge.

    >>> from pyvista import examples
    >>> grid = examples.cells.LagrangeWedge()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([40,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
           33, 34, 35, 36, 37, 38, 39])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.        , 0.        , 1.        ],
                     [1.        , 0.        , 1.        ],
                     [0.        , 1.        , 1.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [0.66666669, 0.33333334, 0.        ],
                     [0.33333334, 0.66666669, 0.        ],
                     [0.        , 0.66666669, 0.        ],
                     [0.        , 0.33333334, 0.        ],
                     [0.33333334, 0.        , 1.        ],
                     [0.66666669, 0.        , 1.        ],
                     [0.66666669, 0.33333334, 1.        ],
                     [0.33333334, 0.66666669, 1.        ],
                     [0.        , 0.66666669, 1.        ],
                     [0.        , 0.33333334, 1.        ],
                     [0.        , 0.        , 0.33333334],
                     [0.        , 0.        , 0.66666669],
                     [1.        , 0.        , 0.33333334],
                     [1.        , 0.        , 0.66666669],
                     [0.        , 1.        , 0.33333334],
                     [0.        , 1.        , 0.66666669],
                     [0.33333334, 0.33333334, 0.        ],
                     [0.33333334, 0.33333334, 1.        ],
                     [0.33333334, 0.        , 0.33333334],
                     [0.66666669, 0.        , 0.33333334],
                     [0.33333334, 0.        , 0.66666669],
                     [0.66666669, 0.        , 0.66666669],
                     [0.66666669, 0.33333334, 0.33333334],
                     [0.33333334, 0.66666669, 0.33333334],
                     [0.66666669, 0.33333334, 0.66666669],
                     [0.33333334, 0.66666669, 0.66666669],
                     [0.        , 0.66666669, 0.33333334],
                     [0.        , 0.33333334, 0.33333334],
                     [0.        , 0.66666669, 0.66666669],
                     [0.        , 0.33333334, 0.66666669],
                     [0.33333334, 0.33333334, 0.33333334],
                     [0.33333334, 0.33333334, 0.66666669]])

    >>> grid.celltypes  # same as pyvista.CellType.LAGRANGE_WEDGE
    array([73], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid', _vtkCellTypeSource(CellType.LAGRANGE_WEDGE, cell_order=cell_order)
    )


def BezierCurve(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Bezier curve.

    This cell corresponds to the :attr:`pyvista.CellType.BEZIER_CURVE` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Bezier curve.

    Examples
    --------
    Create and plot a single Bezier curve.

    >>> from pyvista import examples
    >>> grid = examples.cells.BezierCurve()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([4, 0, 1, 2, 3])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.BEZIER_CURVE
    array([75], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid', _vtkCellTypeSource(CellType.BEZIER_CURVE, cell_order=cell_order)
    )


def BezierTriangle(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Bezier triangle.

    This cell corresponds to the :attr:`pyvista.CellType.BEZIER_TRIANGLE` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Bezier triangle.

    Examples
    --------
    Create and plot a single Bezier triangle.

    >>> from pyvista import examples
    >>> grid = examples.cells.BezierTriangle()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [0.66666669, 0.33333334, 0.        ],
                     [0.33333334, 0.66666669, 0.        ],
                     [0.        , 0.66666669, 0.        ],
                     [0.        , 0.33333334, 0.        ],
                     [0.33333334, 0.33333334, 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.BEZIER_TRIANGLE
    array([76], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid', _vtkCellTypeSource(CellType.BEZIER_TRIANGLE, cell_order=cell_order)
    )


def BezierQuadrilateral(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Bezier quadrilateral.

    This cell corresponds to the :attr:`pyvista.CellType.BEZIER_QUADRILATERAL` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Bezier quadrilateral.

    Examples
    --------
    Create and plot a single Bezier quadrilateral.

    >>> from pyvista import examples
    >>> grid = examples.cells.BezierQuadrilateral()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([16,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [1.        , 1.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [1.        , 0.33333334, 0.        ],
                     [1.        , 0.66666669, 0.        ],
                     [0.33333334, 1.        , 0.        ],
                     [0.66666669, 1.        , 0.        ],
                     [0.        , 0.33333334, 0.        ],
                     [0.        , 0.66666669, 0.        ],
                     [0.33333334, 0.33333334, 0.        ],
                     [0.66666669, 0.33333334, 0.        ],
                     [0.33333334, 0.66666669, 0.        ],
                     [0.66666669, 0.66666669, 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.BEZIER_QUADRILATERAL
    array([77], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid',
        _vtkCellTypeSource(CellType.BEZIER_QUADRILATERAL, cell_order=cell_order),
    )


def BezierTetrahedron(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Bezier tetrahedron.

    This cell corresponds to the :attr:`pyvista.CellType.BEZIER_TETRAHEDRON` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Bezier tetrahedron.

    Examples
    --------
    Create and plot a single Bezier tetrahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.BezierTetrahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([20,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [1.        , 1.        , 0.        ],
                     [0.5       , 0.5       , 0.5       ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [1.        , 0.33333334, 0.        ],
                     [1.        , 0.66666669, 0.        ],
                     [0.66666669, 0.66666669, 0.        ],
                     [0.33333334, 0.33333334, 0.        ],
                     [0.16666667, 0.16666667, 0.16666667],
                     [0.33333334, 0.33333334, 0.33333334],
                     [0.83333331, 0.16666667, 0.16666667],
                     [0.66666669, 0.33333334, 0.33333334],
                     [0.83333331, 0.83333331, 0.16666667],
                     [0.66666669, 0.66666669, 0.33333334],
                     [0.5       , 0.16666667, 0.16666667],
                     [0.83333331, 0.5       , 0.16666667],
                     [0.5       , 0.5       , 0.16666667],
                     [0.66666669, 0.33333334, 0.        ]])

    >>> grid.celltypes  # same as pyvista.CellType.BEZIER_TETRAHEDRON
    array([78], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid',
        _vtkCellTypeSource(CellType.BEZIER_TETRAHEDRON, cell_order=cell_order),
    )


def BezierHexahedron(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Bezier hexahedron.

    This cell corresponds to the :attr:`pyvista.CellType.BEZIER_HEXAHEDRON` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Bezier hexahedron.

    Examples
    --------
    Create and plot a single Bezier hexahedron.

    >>> from pyvista import examples
    >>> grid = examples.cells.BezierHexahedron()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
           33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
           50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [1.        , 1.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.        , 0.        , 1.        ],
                     [1.        , 0.        , 1.        ],
                     [1.        , 1.        , 1.        ],
                     [0.        , 1.        , 1.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [1.        , 0.33333334, 0.        ],
                     [1.        , 0.66666669, 0.        ],
                     [0.33333334, 1.        , 0.        ],
                     [0.66666669, 1.        , 0.        ],
                     [0.        , 0.33333334, 0.        ],
                     [0.        , 0.66666669, 0.        ],
                     [0.33333334, 0.        , 1.        ],
                     [0.66666669, 0.        , 1.        ],
                     [1.        , 0.33333334, 1.        ],
                     [1.        , 0.66666669, 1.        ],
                     [0.33333334, 1.        , 1.        ],
                     [0.66666669, 1.        , 1.        ],
                     [0.        , 0.33333334, 1.        ],
                     [0.        , 0.66666669, 1.        ],
                     [0.        , 0.        , 0.33333334],
                     [0.        , 0.        , 0.66666669],
                     [1.        , 0.        , 0.33333334],
                     [1.        , 0.        , 0.66666669],
                     [1.        , 1.        , 0.33333334],
                     [1.        , 1.        , 0.66666669],
                     [0.        , 1.        , 0.33333334],
                     [0.        , 1.        , 0.66666669],
                     [0.        , 0.33333334, 0.33333334],
                     [0.        , 0.66666669, 0.33333334],
                     [0.        , 0.33333334, 0.66666669],
                     [0.        , 0.66666669, 0.66666669],
                     [1.        , 0.33333334, 0.33333334],
                     [1.        , 0.66666669, 0.33333334],
                     [1.        , 0.33333334, 0.66666669],
                     [1.        , 0.66666669, 0.66666669],
                     [0.33333334, 0.        , 0.33333334],
                     [0.66666669, 0.        , 0.33333334],
                     [0.33333334, 0.        , 0.66666669],
                     [0.66666669, 0.        , 0.66666669],
                     [0.33333334, 1.        , 0.33333334],
                     [0.66666669, 1.        , 0.33333334],
                     [0.33333334, 1.        , 0.66666669],
                     [0.66666669, 1.        , 0.66666669],
                     [0.33333334, 0.33333334, 0.        ],
                     [0.66666669, 0.33333334, 0.        ],
                     [0.33333334, 0.66666669, 0.        ],
                     [0.66666669, 0.66666669, 0.        ],
                     [0.33333334, 0.33333334, 1.        ],
                     [0.66666669, 0.33333334, 1.        ],
                     [0.33333334, 0.66666669, 1.        ],
                     [0.66666669, 0.66666669, 1.        ],
                     [0.33333334, 0.33333334, 0.33333334],
                     [0.66666669, 0.33333334, 0.33333334],
                     [0.33333334, 0.66666669, 0.33333334],
                     [0.66666669, 0.66666669, 0.33333334],
                     [0.33333334, 0.33333334, 0.66666669],
                     [0.66666669, 0.33333334, 0.66666669],
                     [0.33333334, 0.66666669, 0.66666669],
                     [0.66666669, 0.66666669, 0.66666669]])

    >>> grid.celltypes  # same as pyvista.CellType.BEZIER_HEXAHEDRON
    array([79], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid',
        _vtkCellTypeSource(CellType.BEZIER_HEXAHEDRON, cell_order=cell_order),
    )


def BezierWedge(*, cell_order: int = 3) -> UnstructuredGrid:
    """Create a :class:`pyvista.UnstructuredGrid` containing a single Bezier wedge.

    This cell corresponds to the :attr:`pyvista.CellType.BEZIER_WEDGE` cell type.

    Parameters
    ----------
    cell_order : int, default: 3
        Order of interpolation to use.

    Returns
    -------
    pyvista.UnstructuredGrid
        UnstructuredGrid containing a single Bezier wedge.

    Examples
    --------
    Create and plot a single Bezier wedge.

    >>> from pyvista import examples
    >>> grid = examples.cells.BezierWedge()
    >>> examples.plot_cell(grid)

    List the grid's cells.

    >>> grid.cells
    array([40,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
           33, 34, 35, 36, 37, 38, 39])

    List the grid's points.

    >>> grid.points
    pyvista_ndarray([[0.        , 0.        , 0.        ],
                     [1.        , 0.        , 0.        ],
                     [0.        , 1.        , 0.        ],
                     [0.        , 0.        , 1.        ],
                     [1.        , 0.        , 1.        ],
                     [0.        , 1.        , 1.        ],
                     [0.33333334, 0.        , 0.        ],
                     [0.66666669, 0.        , 0.        ],
                     [0.66666669, 0.33333334, 0.        ],
                     [0.33333334, 0.66666669, 0.        ],
                     [0.        , 0.66666669, 0.        ],
                     [0.        , 0.33333334, 0.        ],
                     [0.33333334, 0.        , 1.        ],
                     [0.66666669, 0.        , 1.        ],
                     [0.66666669, 0.33333334, 1.        ],
                     [0.33333334, 0.66666669, 1.        ],
                     [0.        , 0.66666669, 1.        ],
                     [0.        , 0.33333334, 1.        ],
                     [0.        , 0.        , 0.33333334],
                     [0.        , 0.        , 0.66666669],
                     [1.        , 0.        , 0.33333334],
                     [1.        , 0.        , 0.66666669],
                     [0.        , 1.        , 0.33333334],
                     [0.        , 1.        , 0.66666669],
                     [0.33333334, 0.33333334, 0.        ],
                     [0.33333334, 0.33333334, 1.        ],
                     [0.33333334, 0.        , 0.33333334],
                     [0.66666669, 0.        , 0.33333334],
                     [0.33333334, 0.        , 0.66666669],
                     [0.66666669, 0.        , 0.66666669],
                     [0.66666669, 0.33333334, 0.33333334],
                     [0.33333334, 0.66666669, 0.33333334],
                     [0.66666669, 0.33333334, 0.66666669],
                     [0.33333334, 0.66666669, 0.66666669],
                     [0.        , 0.66666669, 0.33333334],
                     [0.        , 0.33333334, 0.33333334],
                     [0.        , 0.66666669, 0.66666669],
                     [0.        , 0.33333334, 0.66666669],
                     [0.33333334, 0.33333334, 0.33333334],
                     [0.33333334, 0.33333334, 0.66666669]])

    >>> grid.celltypes  # same as pyvista.CellType.BEZIER_WEDGE
    array([80], dtype=uint8)

    """
    return cast(
        'UnstructuredGrid', _vtkCellTypeSource(CellType.BEZIER_WEDGE, cell_order=cell_order)
    )


def cell_type_source(  # numpydoc ignore=RT01
    cell_types: int | Sequence[int],
    generator: Literal['examples', 'blocks', 'parametric'] = 'examples',
    *,
    block_dimensions: VectorLike[int] | None = None,
    shrink_factor: float | None = None,
    fill_mode: Literal['exact', 'cycle', 'stop'] = 'exact',
    unsupported_action: Literal['squeeze', 'skip', 'warn', 'error'] = 'error',
) -> MultiBlock:
    """Generate a :class:`~pyvista.MultiBlock` mesh comprised of one or more cell types.

    A separate :class:`~pyvista.UnstructuredGrid` block is generated for each input cell type.
    Cell types may be repeated or mixed in any order. By default, all blocks are stacked
    sequentially along the x-axis, though ``block_dimensions`` may be specified to control this.

    Parameters
    ----------
    cell_types : int | sequence[int]
        Cell types to generate.

    generator : 'examples' | 'blocks' | 'parametric', default: 'examples'
        Method for generating cell type blocks.

        - ``'examples'``: generate blocks using examples from :mod:`pyvista.examples.cells`.
        - ``'parametric'``: generate blocks using :vtk:`vtkCell.GetParametricCoords`.
        - ``'blocks'``: generate blocks using :vtk:`vtkCellTypeSource`.

        .. note::

           - ``'examples'`` supports all concrete :class:`cell types <pyvista.CellType>`, but
             the other generators only support a subset.
           - Both ``'examples'`` and ``'parametric'`` only generate a `single` cell per block,
             whereas ``'blocks'`` may generate multiple cells of the same type in order to fill a
             unit block (e.g. two triangles to fill a square, two wedges to fill a cube).

    block_dimensions : VectorLike[int], optional
        Output dimensions of blocks to generate. By default, all blocks are stacked sequentially
        along the x-axis. The dimensions should be compatible with the number of input cell types.
        Use ``fill_mode`` to handle cases where the dimensions are `not` compatible.

    shrink_factor : float, optional
        Shrink each block by applying a scaling factor. By default, no shrink factor is applied,
        and each generated cell type is scaled to fit inside a unit cube.

    fill_mode : 'exact' | 'cycle' | 'stop', default: 'exact'
        Select how to handle mismatched dimensions.

        - ``'exact'``: the number of cell types must match the specified block dimensions exactly.
        - ``'cycle'``: cycle through and duplicate cell types as required to ensure the
          specified block dimensions are completely filled.
        - ``'stop'``: stop iterating when all specified cell types have been generated.

    unsupported_action : 'skip' | 'squeeze'| 'warn' | 'error', default: 'error'
        Select how to handle unsupported cell types.

        - ``'skip'``: Skip generating a block for unsupported cell types. A ``None`` block is
          included instead. This will create a gap in the output.
        - ``'squeeze'``: Similar to ``'skip'``, but no ``None`` block is appended. Since no block
          is included, there are no gaps and the output appears to be "squeezed" together.
        - ``'warn'``: Similar to ``skip``, but a warning is emitted when an unsupported type is
          encountered.
        - ``'error'``: Raise a ValueError when an unsupported cell type is encountered.

    Examples
    --------
    Generate a single :attr:`~pyvista.CellType.TRIANGLE` cell.

    >>> import pyvista as pv
    >>> from pyvista.examples import cells, cell_type_source, plot_cell
    >>> triangle = cell_type_source(pv.CellType.TRIANGLE)
    >>> plot_cell(triangle)

    This is similar to the :func:`~pyvista.examples.cells.Triangle` grid, except it's a
    :class:`~pyvista.MultiBlock` and its bounds are normalized to fit inside a 1x1x1 grid.

    >>> triangle
    MultiBlock (...)
      N Blocks:   1
      X Bounds:   0.000e+00, 1.000e+00
      Y Bounds:   -5.551e-17, 1.000e+00
      Z Bounds:   5.000e-01, 5.000e-01

    Compare to the un-normalized bounds.

    >>> cells.Triangle()
    UnstructuredGrid (...)
      N Cells:    1
      N Points:   3
      X Bounds:   -5.000e-01, 5.000e-01
      Y Bounds:   -2.887e-01, 5.774e-01
      Z Bounds:   0.000e+00, 0.000e+00
      N Arrays:   0

    Use the ``'parametric'`` generator instead.

    >>> triangle = cell_type_source(pv.CellType.TRIANGLE, 'parametric')
    >>> plot_cell(triangle)

    Use the ``'blocks'`` generator instead.

    >>> triangle = cell_type_source(pv.CellType.TRIANGLE, 'blocks')
    >>> plot_cell(triangle)

    Generate multiple cell types. Here we generate all concrete linear 2D cells.

    >>> cell_types = [
    ...     ctype
    ...     for ctype in pv.CellType
    ...     if ctype.dimension == 2 and ctype.is_linear
    ... ]
    >>> cell_types  # doctest: +NORMALIZE_WHITESPACE
    [<CellType.TRIANGLE: 5>,
     <CellType.TRIANGLE_STRIP: 6>,
     <CellType.POLYGON: 7>,
     <CellType.PIXEL: 8>,
     <CellType.QUAD: 9>]

    >>> cell_blocks = cell_type_source(cell_types, shrink_factor=0.8)
    >>> plot_cell(cell_blocks)

    Each block's name matches the name of the cell type.

    >>> cell_blocks.keys()
    ['TRIANGLE', 'TRIANGLE_STRIP', 'POLYGON', 'PIXEL', 'QUAD']

    Generate the same cell types using the parametric generator. This generator does not support
    triangle strip or polygon cells, so we skip these.

    >>> cell_blocks = cell_type_source(
    ...     cell_types, 'parametric', shrink_factor=0.8, unsupported_action='skip'
    ... )
    >>> plot_cell(cell_blocks)

    Use the blocks generator. It also does not support triangle strip, but it does support polygon.
    This time, we squeeze the blocks together instead of skipping them, and do not shrink them.
    This combination generates a continuous grid with no gaps.

    >>> cell_blocks = cell_type_source(
    ...     cell_types, 'blocks', unsupported_action='squeeze'
    ... )
    >>> plot_cell(cell_blocks)

    Generate cell types on a dimensioned grid.

    >>> lines = [pv.CellType.LINE] * 3
    >>> polygons = [pv.CellType.POLYGON] * 3
    >>> wedges = [pv.CellType.WEDGE] * 3
    >>> pyramids = [pv.CellType.PYRAMID] * 3
    >>> cell_types = [*lines, *polygons, *wedges, *pyramids]

    Plot them with 3 cells in the x-direction, and 4 cells in the y-direction.

    >>> cell_blocks = cell_type_source(cell_types, block_dimensions=(3, 4, 1))
    >>> size_kwargs = dict(point_size=40, font_size=20)
    >>> plot_cell(cell_blocks, cpos='xy', **size_kwargs)

    Reverse the x and y dimension.

    >>> cell_blocks = cell_type_source(cell_types, block_dimensions=(4, 3, 1))
    >>> plot_cell(cell_blocks, cpos='xy', **size_kwargs)

    Use the ``'stop'`` fill mode if there is a mismatch between the number of cell types and block
    dimensions. Here, the last two pyramid cell types are omitted.

    >>> cell_blocks = cell_type_source(
    ...     cell_types[:-2], block_dimensions=(3, 4, 1), fill_mode='stop'
    ... )
    >>> plot_cell(cell_blocks, cpos='xy', **size_kwargs)

    Alternatively, cycle through the cell types again to completely fill the dimensions. In this
    case, the line type is reused to fill the gap.

    >>> cell_blocks = cell_type_source(
    ...     cell_types[:-2], block_dimensions=(3, 4, 1), fill_mode='cycle'
    ... )
    >>> plot_cell(cell_blocks, cpos='xy', **size_kwargs)

    Generate a 5x5x5 grid comprised of all 3D cell types with no gaps.

    >>> cell_types = [ctype for ctype in pv.CellType if ctype.dimension == 3]
    >>> cell_blocks = cell_type_source(
    ...     cell_types,
    ...     'blocks',
    ...     block_dimensions=(5, 5, 5),
    ...     unsupported_action='squeeze',
    ...     fill_mode='cycle',
    ... )
    >>> cell_blocks.plot(show_edges=True, opacity=0.5, line_width=3)

    Combine into a single grid and show :attr:`~pyvista.DataSet.distinct_cell_types`.

    >>> cell_blocks.combine().distinct_cell_types  # doctest: +NORMALIZE_WHITESPACE
    {<CellType.TETRA: 10>, <CellType.VOXEL: 11>, <CellType.HEXAHEDRON: 12>, <CellType.WEDGE: 13>,
     <CellType.PYRAMID: 14>, <CellType.PENTAGONAL_PRISM: 15>, <CellType.HEXAGONAL_PRISM: 16>,
     <CellType.QUADRATIC_TETRA: 24>, <CellType.QUADRATIC_HEXAHEDRON: 25>,
     <CellType.QUADRATIC_WEDGE: 26>, <CellType.QUADRATIC_PYRAMID: 27>,
     <CellType.TRIQUADRATIC_HEXAHEDRON: 29>, <CellType.TRIQUADRATIC_PYRAMID: 37>,
     <CellType.POLYHEDRON: 42>, <CellType.LAGRANGE_TETRAHEDRON: 71>,
     <CellType.LAGRANGE_HEXAHEDRON: 72>, <CellType.LAGRANGE_WEDGE: 73>,
     <CellType.BEZIER_TETRAHEDRON: 78>, <CellType.BEZIER_HEXAHEDRON: 79>,
     <CellType.BEZIER_WEDGE: 80>}

    Compare the first 25 cell types from the different generators. Note that some values, e.g.
    ``17``, do not correspond at all to any cell type, so gaps are expected in all outputs.

    >>> kwargs = dict(
    ...     cell_types=range(1, 26),
    ...     block_dimensions=(5, 5, 1),
    ...     unsupported_action='skip',
    ... )
    >>> cell_blocks = cell_type_source(generator='examples', **kwargs)
    >>> plot_cell(cell_blocks, cpos='xy', **size_kwargs)

    >>> cell_blocks = cell_type_source(generator='parametric', **kwargs)
    >>> plot_cell(cell_blocks, cpos='xy', **size_kwargs)

    >>> cell_blocks = cell_type_source(generator='blocks', **kwargs)
    >>> plot_cell(cell_blocks, cpos='xy', **size_kwargs)

    """

    def _examples_grid(cell_type: CellType) -> UnstructuredGrid | None:
        if (example := cell_type._example) is None:
            return None
        return globals()[example]()

    def _parametric_grid(cell_type: CellType) -> UnstructuredGrid | None:
        return _isoparametric_grid(cell_type)

    def _blocks_grid(cell_type: CellType) -> UnstructuredGrid | None:
        return _vtkCellTypeSource(cell_type, cell_order=3, single_cell=False)

    _shrink_factor = (
        1.0
        if shrink_factor is None
        else _validation.validate_number(shrink_factor, must_be_in_range=[0.0, 1.0])
    )
    ctypes: list[int] = []
    for ctype in cell_types if isinstance(cell_types, Iterable) else [cell_types]:
        try:
            ctypes.append(CellType(ctype))
        except ValueError:
            ctypes.append(ctype)

    if block_dimensions is None:
        dimension: int | VectorLike[int] = (len(ctypes), 1, 1)
    else:
        requested_size = np.prod(block_dimensions)
        actual_size = len(ctypes)
        if requested_size < actual_size:
            msg = (
                f'Requested dimension {block_dimensions} is too small. '
                f'Number of cell types to generate ({actual_size}) exceeds '
                f'the number of blocks requested ({requested_size}).'
            )
            raise ValueError(msg)
        elif requested_size > actual_size:
            if fill_mode == 'exact':
                msg = (
                    f'Requested dimension {block_dimensions} is too large. '
                    f'Number of cell types to generate ({actual_size}) is less than '
                    f'the number of blocks requested ({requested_size}).\n'
                    f'Use `fill_mode` to prevent an error from being raised.'
                )
                raise ValueError(msg)
        dimension = block_dimensions
    dims = _validation.validate_array3(dimension, name='block_dimensions')
    cell_centers = pv.ImageData(dimensions=dims + 1).cell_centers().points

    if generator == 'examples':
        generate_grid = _examples_grid
    elif generator == 'parametric':
        generate_grid = _parametric_grid
    elif generator == 'blocks':
        generate_grid = _blocks_grid

    # Generate mesh for each cell type
    distinct_meshes = pv.MultiBlock()
    for ctype in ctypes:
        if isinstance(ctype, CellType):
            grid = generate_grid(ctype)
            name = ctype.name
        else:
            grid = None
            name = 'None'
        if grid is None:
            if unsupported_action in ['warn', 'error']:
                invalid = ' ' if isinstance(ctype, CellType) else ' is not a valid cell type and '
                msg = f'{ctype!r}{invalid}is not supported by the {generator!r} generator.'
                if unsupported_action == 'error':
                    raise ValueError(msg)
                warn_external(msg)
        elif grid.bounds_size != (1.0, 1.0, 1.0) or _shrink_factor != 1.0:
            grid = grid.resize(bounds_size=_shrink_factor)
        distinct_meshes.append(grid, name)

    # Build output from distinct meshes
    output = pv.MultiBlock()
    iterator = distinct_meshes.recursive_iterator('items')
    iterator = itertools.cycle(iterator) if fill_mode == 'cycle' else iterator

    center_iter = iter(cell_centers)
    name_counts: dict[str, int] = {}
    for name, block in iterator:
        if block is None:
            if unsupported_action == 'squeeze':
                continue  # preserve center
            next(center_iter, None)  # consume center
            continue

        center = next(center_iter, None)
        if center is None:
            break

        # Ensure unique names
        count = name_counts.get(name, 0)
        block_name = name if count == 0 else f'{name}_{count}'
        name_counts[name] = count + 1

        # Ensure block is independent
        block_mesh = block.copy() if block in output else block
        # Position block on the grid
        block_mesh.center = center
        output[block_name] = block_mesh
    return output


def _isoparametric_grid(celltype: CellType) -> UnstructuredGrid | None:
    if celltype.vtk_class is None or celltype in _NOT_SUPPORTED_PARAMETRIC:
        return None
    cell = celltype.vtk_class()

    # Create points
    cell.Initialize()
    pcoords = cell.GetParametricCoords()
    points = np.zeros((cell.GetNumberOfPoints(), 3))
    for i in range(len(points)):
        points[i] = (pcoords[3 * i], pcoords[3 * i + 1], pcoords[3 * i + 2])

    cells = [len(points), *list(range(len(points)))]
    return UnstructuredGrid(cells, [celltype], points)


def _vtkCellTypeSource(
    celltype: CellType, *, cell_order: int = 3, single_cell: bool = True
) -> UnstructuredGrid | None:
    """Use vtkCellTypeSource to generate UnstructuredGrid with a single cell type."""
    if celltype.vtk_class is None or celltype in _NOT_SUPPORTED_CELL_SOURCE:
        return None
    src = _vtk.vtkCellTypeSource()
    src.SetBlocksDimensions(1, 1, 1)
    src.SetCellOrder(cell_order)
    src.SetCellType(celltype)
    src.Update()
    ugrid = cast('_vtk.vtkUnstructuredGrid', src.GetOutput())

    if single_cell:
        cell0 = ugrid.GetCell(0)
        ugrid = _vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(cell0.GetPoints())
        ids = _vtk.vtkIdList()
        for i in range(cell0.GetNumberOfPoints()):
            ids.InsertNextId(i)
        ugrid.InsertNextCell(cell0.GetCellType(), ids)

    return pv.wrap(ugrid)
