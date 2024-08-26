"""Define types of cells."""

from __future__ import annotations

from enum import IntEnum
import textwrap
from typing import Literal
from typing import NamedTuple

from . import _vtk_core as _vtk

_DROPDOWN_TEMPLATE = """
.. dropdown:: More info
    :icon: info

{}
"""

_GRID_TEMPLATE_NO_IMAGE = """
.. grid:: 1
    :margin: 1

    .. grid-item::

        {}{}{}
"""

_GRID_TEMPLATE_WITH_IMAGE = """
.. grid:: 1 2 2 2
    :reverse:
    :margin: 1

    .. grid-item::
        :columns: 12 4 4 4

        .. card::
            :class-body: sd-px-0 sd-py-0 sd-rounded-3

            .. image:: /../_build/plot_directive/api/examples/_autosummary/pyvista-examples-cells-{}-1_00_00.png

    .. grid-item::
        :columns: 12 8 8 8

        {}{}{}
"""


def _indent_paragraph(string, level):
    indentation = "".join(['    '] * level)
    return textwrap.indent(textwrap.dedent(string).strip(), indentation)


def _generate_linearity_badge(is_linear: bool):
    LINEAR_BADGE = ':bdg-primary:`Linear`'
    NON_LINEAR_BADGE = ':bdg-success:`Non-linear`'
    return LINEAR_BADGE if is_linear else NON_LINEAR_BADGE


def _generate_dimension_badge(dimension: int):
    return f':bdg-secondary:`{dimension}D`'


def _generate_points_badge(num_points: int):
    return f':bdg-muted-line:`Points: {num_points}`'


def _generate_edges_badge(num_edges: int):
    return f':bdg-muted-line:`Edges: {num_edges}`'


def _generate_faces_badge(num_faces: int):
    return f':bdg-muted-line:`Faces: {num_faces}`'


class _CellTypeTuple(NamedTuple):
    value: int
    cell_class: type[_vtk.vtkCell] | None = None
    short_doc: str | None = None
    long_doc: str | None = None
    example: str | None = None
    points_override: Literal['variable', 'n/a'] | None = None
    edges_override: Literal['variable', 'n/a'] | None = None
    faces_override: Literal['variable', 'n/a'] | None = None


class _DocIntEnum(IntEnum):
    """Enable documentation for enum members."""

    def __new__(
        cls,
        value,
        _cell_class: type[_vtk.vtkCell] | None = None,
        _short_doc: str | None = None,
        _long_doc: str | None = None,
        _example: str | None = None,
        _points_override: Literal['variable', 'n/a'] | None = None,
        _edges_override: Literal['variable', 'n/a'] | None = None,
        _faces_override: Literal['variable', 'n/a'] | None = None,
    ):
        """Create new enum.

        Optionally specify documentation info.

        Parameters
        ----------
        value : int
            Integer value of the cell type.

        _cell_class : type[_vtk.vtkCell], optional
            VTK class for this cell type.

        _short_doc : str, optional
            Short description of this cell type. Typically a single line but no more
            than 3-4 lines. Should only include a general description and no technical
            details or cell connectivity.

        _long_doc : str, optional
            Long description of this cell type. This information is hidden inside a
            drop-down. Include cell connectivity here or any other technical details.

        _example : str, optional
            Name of the example for this cell type in `pyvista.examples.cell.<NAME>`.
            When specified, the first figure from this example is used as the image for
            the cell.

        _points_override: 'variable' | 'n/a', optional
            Override the value shown for this cell type's `Points` badge. May be
            useful for composite cells (e.g. POLY_LINE or POLY_VERTEX) where a value
            of ``0`` may otherwise be shown. By default, the value from ``cell_class``
            is used.

        _edges_override: 'variable' | 'n/a', optional
            Override the value shown for this cell type's `Edges` badge. May be
            useful for composite cells (e.g. POLY_LINE or POLY_VERTEX) where a value
            of ``0`` may otherwise be shown. By default, the value from ``cell_class``
            is used.

        _faces_override: 'variable' | 'n/a', optional
            Override the value shown for this cell type's `Faces` badge. May be
            useful for composite cells (e.g. POLY_LINE or POLY_VERTEX) where a value
            of ``0`` may otherwise be shown. By default, the value from ``cell_class``
            is used.
        """
        self = int.__new__(cls, value)
        self._value_ = value
        self.__doc__ = ''

        # Generate cell type documentation if specified
        if _cell_class or _short_doc or _long_doc or _example:
            if _cell_class:
                cell = _cell_class()
                linearity_badge = _generate_linearity_badge(cell.IsLinear())
                dimension_badge = _generate_dimension_badge(cell.GetCellDimension())

                points = _points_override if _points_override else cell.GetNumberOfPoints()
                points_badge = _generate_points_badge(points)

                edges = _edges_override if _edges_override else cell.GetNumberOfEdges()
                edges_badge = _generate_edges_badge(edges)

                faces = _faces_override if _faces_override else cell.GetNumberOfFaces()
                faces_badge = _generate_faces_badge(faces)

                badges = f'{linearity_badge} {dimension_badge} {points_badge} {edges_badge} {faces_badge}\n\n'
            else:
                badges = ''

            _short_doc = '' if _short_doc is None else _indent_paragraph(_short_doc, level=2)

            _long_doc = (
                ''
                if _long_doc is None
                else _indent_paragraph(
                    _DROPDOWN_TEMPLATE.format(_indent_paragraph(_long_doc, level=1)), level=2
                )
            )
            if _short_doc and _long_doc:
                _short_doc += '\n\n'

            self.__doc__ += (
                _GRID_TEMPLATE_NO_IMAGE.format(badges, _short_doc, _long_doc)
                if _example is None
                else _GRID_TEMPLATE_WITH_IMAGE.format(_example, badges, _short_doc, _long_doc)
            )

        return self


class CellType(_DocIntEnum):
    """Define types of cells.

    Notes
    -----
    See `vtkCellType.h
    <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_ for all
    cell types.

    Examples
    --------
    Create a single cube. Notice how the cell type is defined using the
    ``CellType``.

    >>> import numpy as np
    >>> from pyvista import CellType
    >>> import pyvista as pv
    >>> cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> cell_type = np.array([CellType.HEXAHEDRON], np.int8)
    >>> points = np.array(
    ...     [
    ...         [0, 0, 0],
    ...         [1, 0, 0],
    ...         [1, 1, 0],
    ...         [0, 1, 0],
    ...         [0, 0, 1],
    ...         [1, 0, 1],
    ...         [1, 1, 1],
    ...         [0, 1, 1],
    ...     ],
    ...     dtype=np.float32,
    ... )
    >>> grid = pv.UnstructuredGrid(cells, cell_type, points)
    >>> grid
    UnstructuredGrid (...)
      N Cells:    1
      N Points:   8
      X Bounds:   0.000e+00, 1.000e+00
      Y Bounds:   0.000e+00, 1.000e+00
      Z Bounds:   0.000e+00, 1.000e+00
      N Arrays:   0

    """

    ####################################################################################
    # Linear cells
    EMPTY_CELL = _CellTypeTuple(
        value=_vtk.VTK_EMPTY_CELL,
        cell_class=_vtk.vtkEmptyCell,
        short_doc="""Used as a place-holder during processing.""",
    )
    VERTEX = _CellTypeTuple(
        value=_vtk.VTK_VERTEX,
        cell_class=_vtk.vtkVertex,
        example="Vertex",
        short_doc="""Represent a point in 3D space.""",
    )
    POLY_VERTEX = _CellTypeTuple(
        value=_vtk.VTK_POLY_VERTEX,
        cell_class=_vtk.vtkPolyVertex,
        example="PolyVertex",
        points_override='variable',
        short_doc="""Represent a set of points in 3D space.""",
    )
    LINE = _CellTypeTuple(
        value=_vtk.VTK_LINE,
        cell_class=_vtk.vtkLine,
        example="Line",
        short_doc="""Represent a 1D line.""",
    )
    POLY_LINE = _CellTypeTuple(
        value=_vtk.VTK_POLY_LINE,
        cell_class=_vtk.vtkPolyLine,
        example="PolyLine",
        points_override='variable',
        short_doc="""Represent a set of 1D lines.""",
    )
    TRIANGLE = _CellTypeTuple(
        value=_vtk.VTK_TRIANGLE,
        cell_class=_vtk.vtkTriangle,
        example="Triangle",
        short_doc="""Represent a 2D triangle.""",
    )
    TRIANGLE_STRIP = _CellTypeTuple(
        value=_vtk.VTK_TRIANGLE_STRIP,
        cell_class=_vtk.vtkTriangleStrip,
        example="TriangleStrip",
        points_override='variable',
        edges_override='variable',
        short_doc="""
        Represent a 2D triangle strip.

        A triangle strip is a compact representation of triangles connected
        edge-to-edge in strip fashion.
        """,
        long_doc="""
        The connectivity of a triangle strip is three points defining an
        initial triangle, then for each additional triangle, a single point that,
        combined with the previous two points, defines the next triangle.
        """,
    )
    POLYGON = _CellTypeTuple(
        value=_vtk.VTK_POLYGON,
        cell_class=_vtk.vtkPolygon,
        example="Polygon",
        points_override='variable',
        edges_override='variable',
        short_doc="""
        Represent a 2D n-sided polygon.

        The polygons cannot have any internal holes, and cannot self-intersect.
        """,
        long_doc="""
        Define the polygon with n-points ordered in the counter-clockwise
        direction. Do not repeat the last point.
        """,
    )
    PIXEL = _CellTypeTuple(
        value=_vtk.VTK_PIXEL,
        cell_class=_vtk.vtkPixel,
        example='Pixel',
        short_doc="""
        Represents a 2D orthogonal quadrilateral.

        Unlike ``QUAD`` cells, the corners are at right angles, and aligned along
        x-y-z coordinate axes.
        """,
    )
    QUAD = _CellTypeTuple(
        value=_vtk.VTK_QUAD,
        cell_class=_vtk.vtkQuad,
        example="Quadrilateral",
        short_doc="""
        Represent a 2D quadrilateral.

        It is defined by the four points ``(0,1,2,3)`` in counterclockwise order.
        """,
    )
    TETRA = _CellTypeTuple(
        value=_vtk.VTK_TETRA,
        cell_class=_vtk.vtkTetra,
        example="Tetrahedron",
        short_doc="""Represents a 3D tetrahedron.""",
        long_doc="""
        The tetrahedron is defined by the four points ``(0-3)`` where ``(0,1,2)``
        is the base of the tetrahedron which, using the right hand rule, forms a
        triangle whose normal points in the direction of the fourth point.
        """,
    )
    VOXEL = _CellTypeTuple(
        value=_vtk.VTK_VOXEL,
        cell_class=_vtk.vtkVoxel,
        example="Voxel",
        short_doc="""
        Represents a 3D orthogonal parallelepiped.

        Unlike ``HEXAHEDRON``, ``VOXEL`` has interior angles of 90 degrees, and its
        sides are parallel to the coordinate axes.
        """,
    )
    HEXAHEDRON = _CellTypeTuple(
        value=_vtk.VTK_HEXAHEDRON,
        cell_class=_vtk.vtkHexahedron,
        example="Hexahedron",
        short_doc="""Represent a 3D rectangular hexahedron.""",
        long_doc="""
        The hexahedron is defined by the eight points ``(0-7)`` where ``(0,1,2,3)``
        is the base of the hexahedron which, using the right hand rule, forms a
        quadrilateral whose normal points in the direction of the opposite face
        ``(4,5,6,7)``.
        """,
    )
    WEDGE = _CellTypeTuple(
        value=_vtk.VTK_WEDGE,
        cell_class=_vtk.vtkWedge,
        example="Wedge",
        short_doc="""
        Represent a linear 3D wedge.

        A wedge consists of two triangular and three quadrilateral faces.
        """,
        long_doc="""
        The cell is defined by the six points ``(0-5)`` where ``(0,1,2)`` is the
        base of the wedge which, using the right hand rule, forms a triangle whose
        normal points outward (away from the triangular face ``(3,4,5)``).
        """,
    )
    PYRAMID = _CellTypeTuple(
        value=_vtk.VTK_PYRAMID,
        cell_class=_vtk.vtkPyramid,
        example="Pyramid",
        short_doc="""
        Represent a 3D pyramid.

        A pyramid consists of a rectangular base with four triangular faces.
        """,
        long_doc="""
        The pyramid is defined by the five points ``(0-4)`` where ``(0,1,2,3)`` is
        the base of the pyramid which, using the right hand rule, forms a
        quadrilateral whose normal points in the direction of the pyramid apex at
        vertex ``(4)``. The parametric location of vertex ``(4)`` is ``[0, 0, 1]``.
        """,
    )
    PENTAGONAL_PRISM = _CellTypeTuple(
        value=_vtk.VTK_PENTAGONAL_PRISM,
        cell_class=_vtk.vtkPentagonalPrism,
        example="PentagonalPrism",
        short_doc="""Represent a convex 3D prism with a pentagonal base.""",
        long_doc="""
        The prism is defined by the ten points ``(0-9)``, where ``(0,1,2,3,4)`` is
        the base of the prism which, using the right hand rule, forms a pentagon
        whose normal points is in the direction of the opposite face ``(5,6,7,8,9)``.
        """,
    )
    HEXAGONAL_PRISM = _CellTypeTuple(
        value=_vtk.VTK_HEXAGONAL_PRISM,
        cell_class=_vtk.vtkHexagonalPrism,
        example="HexagonalPrism",
        short_doc="""Represent a 3D prism with hexagonal base.""",
        long_doc="""
        The prism is defined by the twelve points ``(0-11)`` where (0,1,2,3,4,5)
        is the base of the prism which, using the right hand rule, forms a hexagon
        whose normal points is in the direction of the opposite face ``(6,7,8,9,10,11)``.
        """,
    )
    ####################################################################################
    # Quadratic, isoparametric cells
    QUADRATIC_EDGE = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_EDGE,
        cell_class=_vtk.vtkQuadraticEdge,
        short_doc="""
        Represent a 1D, 3-node, iso-parametric parabolic line.

        The cell includes a mid-edge node.
        """,
        long_doc="""
        The ordering of the three points defining the cell is point ids
        ``(0,1,2)`` where id ``(2)`` is the mid-edge node.
        """,
    )
    QUADRATIC_TRIANGLE = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_TRIANGLE,
        cell_class=_vtk.vtkQuadraticTriangle,
        short_doc="""
        Represent a 2D, 6-node, iso-parametric parabolic triangle.

        The cell includes a mid-edge node for each of the three edges of the cell.
        """,
        long_doc="""
        The ordering of the six points defining the cell is point ids
        ``(0-2, 3-5)`` where:

        - id ``(3)`` is the mid-edge node between points ``(0,1)``.
        - id ``(4)`` is the mid-edge node between points ``(1,2)``.
        - id ``(5)`` is the mid-edge node between points ``(2,0)``.

        """,
    )
    QUADRATIC_QUAD = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_QUAD,
        cell_class=_vtk.vtkQuadraticQuad,
        short_doc="""
        Represent a 2D, 8-node iso-parametric parabolic quadrilateral element.

        The cell includes a mid-edge node for each of the four edges of the cell.
        """,
        long_doc="""
        The ordering of the eight points defining the cell are point ids
        ``(0-3, 4-7)`` where:

        - ids ``(0-3)`` define the four corner vertices of the quad.
        - ids ``(4-7)`` define the mid-edge nodes ``(0,1)``, ``(1,2)``, ``(2,3)``, ``(3,0)``.

        """,
    )
    QUADRATIC_POLYGON = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_POLYGON,
        cell_class=_vtk.vtkQuadraticPolygon,
        points_override='variable',
        edges_override='variable',
        short_doc="""
        Represent a 2D n-sided (2*n nodes) parabolic polygon.

        The polygon cannot have any internal holes, and cannot self-intersect.
        The cell includes a mid-edge node for each of the n edges of the cell.
        """,
        long_doc="""
        The ordering of the 2*n points defining the cell are point ids
        ``(0..n-1, n..2*n-1)`` where:

        - ids ``(0..n-1)`` define the corner vertices of the polygon.
        - ids ``(n..2*n-1)`` define the mid-edge nodes.

        Define the polygon with points ordered in the counter-clockwise direction.
        Do not repeat the last point.
        """,
    )
    QUADRATIC_TETRA = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_TETRA,
        cell_class=_vtk.vtkQuadraticTetra,
        short_doc="""
        Represent a 3D, 10-node, iso-parametric parabolic tetrahedron.

        The cell includes a mid-edge node on each of the side edges of the tetrahedron.
        """,
        long_doc="""
        The ordering of the ten points defining the cell is point ids ``(0-3, 4-9)``
        where:

        - ids ``(0-3)`` are the four tetra vertices.
        - ids ``(4-9)`` are the mid-edge nodes between ``(0,1)``, ``(1,2)``, ``(2,0)``,
          ``(0,3)``, ``(1,3)``, and ``(2,3)``.

        """,
    )
    QUADRATIC_HEXAHEDRON = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_HEXAHEDRON,
        cell_class=_vtk.vtkQuadraticHexahedron,
        short_doc="""
        Represent a 3D, 20-node iso-parametric parabolic hexahedron.

        The cell includes a mid-edge node.
        """,
        long_doc="""
        The ordering of the twenty points defining the cell is point ids
        ``(0-7, 8-19)`` where:

        - ids ``(0-7)`` are the eight corner vertices of the cube.
        - ids ``(8-19)`` are the twelve mid-edge nodes.

        The mid-edge nodes lie on the edges defined by ``(0,1)``, ``(1,2)``, ``(2,3)``,
        ``(3,0)``, ``(4,5)``, ``(5,6)``, ``(6,7)``, ``(7,4)``, ``(0,4)``, ``(1,5)``,
        ``(2,6)``, ``(3,7)``.
        """,
    )
    QUADRATIC_WEDGE = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_WEDGE,
        cell_class=_vtk.vtkQuadraticWedge,
        short_doc="""
        Represent a 3D, 15-node iso-parametric parabolic wedge.

        The cell includes a mid-edge node.
        """,
        long_doc="""
        The ordering of the fifteen points defining the cell is point ids
        ``(0-5, 6-14)`` where:

        - ids ``(0-5)`` are the six corner vertices of the wedge, defined analogously to
          the six points in ``WEDGE`` (points ``(0,1,2)`` form the base of the wedge
          which, using the right hand rule, forms a triangle whose normal points
          away from the triangular face ``(3,4,5)``).
        - ids ``(6-14)`` are the nine mid-edge nodes.

        The mid-edge nodes lie on the edges defined by ``(0,1)``, ``(1,2)``, ``(2,0)``,
        ``(3,4)``, ``(4,5)``, ``(5,3)``, ``(0,3)``, ``(1,4)``, ``(2,5)``.
        """,
    )
    QUADRATIC_PYRAMID = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_PYRAMID,
        cell_class=_vtk.vtkQuadraticPyramid,
        short_doc="""
        Represent a 3D, 13-node iso-parametric parabolic pyramid.

        The cell includes a mid-edge node.
        """,
        long_doc="""
        The ordering of the thirteen points defining the cell is point ids
        ``(0-4, 5-12)`` where:

        - ids ``(0-4)`` are the five corner vertices of the pyramid
        - ids ``(5-12)`` are the eight mid-edge nodes.

        The mid-edge nodes lie on the edges defined by ``(0,1)``, ``(1,2)``, ``(2,3)``,
        ``(3,0)``, ``(0,4)``, ``(1,4)``, ``(2,4)``, ``(3,4)``, respectively.
        The parametric location of vertex ``(4)`` is ``[0, 0, 1]``.
        """,
    )
    BIQUADRATIC_QUAD = _CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUAD,
        cell_class=_vtk.vtkBiQuadraticQuad,
        short_doc="""
        Represent a 2D, 9-node iso-parametric parabolic quadrilateral element with a center-point.

        The cell includes a mid-edge node for each of the four edges of the cell and
        a center node at the surface.
        """,
        long_doc="""
        The ordering of the eight points defining the cell are point ids
        ``(0-3, 4-8)`` where:

        - ids ``(0-3)`` define the four corner vertices of the quad.
        - ids ``(4-7)`` define the mid-edge nodes ``(0,1)``, ``(1,2)``, ``(2,3)``, ``(3,0)``.
        - id ``(8)`` defines the face center node.

        """,
    )
    TRIQUADRATIC_HEXAHEDRON = _CellTypeTuple(
        value=_vtk.VTK_TRIQUADRATIC_HEXAHEDRON,
        cell_class=_vtk.vtkTriQuadraticHexahedron,
        short_doc="""
        Represent a 3D, 27-node iso-parametric triquadratic hexahedron.

        The cell includes 8 edge nodes, 12 mid-edge nodes, 6 mid-face nodes and one
        mid-volume node.
        """,
        long_doc="""
        The ordering of the 27 points defining the cell is point ids
        ``(0-7, 8-19, 20-25, 26)`` where:

        - ids ``(0-7)`` are the eight corner vertices of the cube.
        - ids ``(8-19)`` are the twelve mid-edge nodes.
        - ids ``(20-25)`` are the six mid-face nodes.
        - id ``(26)`` is the mid-volume node.

        The mid-edge nodes lie on the edges defined by ``(0,1)``, ``(1,2)``, ``(2,3)``,
        ``(3,0)``, ``(4,5)``, ``(5,6)``, ``(6,7)``, ``(7,4)``, ``(0,4)``, ``(1,5)``,
        ``(2,6)``, ``(3,7)``.

        The mid-surface nodes lies on the faces defined by (first edge nodes
        ids, then mid-edge nodes ids):

        - ``(0,1,5,4; 8,17,12,16)``
        - ``(1,2,6,5; 9,18,13,17)``
        - ``(2,3,7,6, 10,19,14,18)``
        - ``(3,0,4,7; 11,16,15,19)``
        - ``(0,1,2,3; 8,9,10,11)``
        - ``(4,5,6,7; 12,13,14,15)``

        The last point lies in the center of the cell ``(0,1,2,3,4,5,6,7)``.
        """,
    )
    if hasattr(_vtk, "VTK_TRIQUADRATIC_PYRAMID"):
        TRIQUADRATIC_PYRAMID = _CellTypeTuple(
            value=_vtk.VTK_TRIQUADRATIC_PYRAMID,
            cell_class=_vtk.vtkTriQuadraticPyramid,
            short_doc="""
            Represent a second order 3D iso-parametric 19-node pyramid.

            The cell includes 5 corner nodes, 8 mid-edge nodes, 5 mid-face nodes,
            and 1 volumetric centroid node.
            """,
            long_doc="""
            The ordering of the nineteen points defining the cell is point
            ids ``(0-4, 5-12, 13-17, 18)``, where:

            - ids ``(0-4)`` are the five corner vertices of the pyramid.
            - ids ``(5-12)`` are the 8 mid-edge nodes.
            - ids ``(13-17)`` are the 5 mid-face nodes.
            - id ``(19)`` is the volumetric centroid node.

            The mid-edge nodes lie on the edges defined by ``(0, 1)``, ``(1, 2)``,
            ``(2, 3)``, ``(3, 0)``, ``(0, 4)``, ``(1, 4)``, ``(2, 4)``, ``(3, 4)``,
            respectively.

            The mid-face nodes lie on the faces defined by (first corner nodes ids,
            then mid-edge node ids):

            - quadrilateral face: ``(0,3,2,1; 8,7,6,5)``
            - triangle face 1: ``(0,1,4; 5,10,9)``
            - triangle face 2: ``(1,2,4; 6,11,10)``
            - triangle face 3: ``(2,3,4; 7,12,11)``
            - triangle face 5: ``(3,0,4; 8,9,12)``

            The last point lies in the center of the cell ``(0,1,2,3,4)``.
            The parametric location of vertex ``(4)`` is ``[0.5, 0.5, 1]``.
            """,
        )
    QUADRATIC_LINEAR_QUAD = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_LINEAR_QUAD,
        cell_class=_vtk.vtkQuadraticLinearQuad,
        short_doc="""
        Represent a 2D, 6-node iso-parametric quadratic-linear quadrilateral element.

        The cell includes a mid-edge node for two of the four edges.
        """,
        long_doc="""
        The ordering of the six points defining the cell are point ids
        ``(0-3, 4-5)`` where:

        - ids ``(0-3)`` define the four corner vertices of the quad.
        - ids ``(4-7)`` define the mid-edge nodes ``(0,1)`` and ``(2,3)``.

        """,
    )
    QUADRATIC_LINEAR_WEDGE = _CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_LINEAR_WEDGE,
        cell_class=_vtk.vtkQuadraticLinearWedge,
        short_doc="""
        Represent a 3D, 12-node iso-parametric linear quadratic wedge.

        The cell includes mid-edge node in the triangle edges.
        """,
        long_doc="""
        The ordering of the 12 points defining the cell is point ids
        ``(0-5, 6-12)`` where:

        - ids ``(0-5`` are the six corner vertices of the wedge.
        - ids ``(6-12)`` are the six mid-edge nodes.

        The mid-edge nodes lie on the edges defined by ``(0,1)``, ``(1,2)``, ``(2,0)``,
        ``(3,4)``, ``(4,5)``, ``(5,3)``.
        The edges ``(0,3)``, ``(1,4)``, ``(2,5)`` don't have mid-edge nodes.
        """,
    )
    BIQUADRATIC_QUADRATIC_WEDGE = _CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUADRATIC_WEDGE,
        cell_class=_vtk.vtkBiQuadraticQuadraticWedge,
        short_doc="""
        Represent a 3D, 18-node iso-parametric bi-quadratic wedge.

        The cell includes a mid-edge node.
        """,
        long_doc="""
        The ordering of the 18 points defining the cell is point ids
        ``(0-5, 6-15, 16-18)`` where:

        - ids ``(0-5)`` are the six corner vertices of the wedge.
        - ids ``(6-15)`` are the nine mid-edge nodes.
        - ids ``(16-18)`` are the three center-face nodes.

        The mid-edge nodes lie on the edges defined by ``(0,1)``, ``(1,2)``, ``(2,0)``,
        ``(3,4)``, ``(4,5)``, ``(5,3)``, ``(0,3)``, ``(1,4)``, ``(2,5)``.

        The center-face nodes are lie in quads ``16-(0,1,4,3)``, ``17-(1,2,5,4)`` and
        ``18-(2,0,3,5)``.
        """,
    )
    BIQUADRATIC_QUADRATIC_HEXAHEDRON = _CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON,
        cell_class=_vtk.vtkBiQuadraticQuadraticHexahedron,
        short_doc="""
        Represent a 3D, 24-node iso-parametric biquadratic hexahedron.

        The cell includes mid-edge and center-face nodes.
        """,
        long_doc="""
        The ordering of the 24 points defining the cell is point ids
        ``(0-7, 8-19, 20-23)`` where:

        - ids ``(0-7)`` are the eight corner vertices of the cube.
        - ids ``(8-19)`` are the twelve mid-edge nodes.
        - ids ``(20-23)`` are the center-face nodes.

        The mid-edge nodes lie on the edges defined by ``(0,1)``, ``(1,2)``, ``(2,3)``,
        ``(3,0)``, ``(4,5)``, ``(5,6)``, ``(6,7)``, ``(7,4)``, ``(0,4)``, ``(1,5)``,
        ``(2,6)``, ``(3,7)``.

        The center face nodes lie in quads ``22-(0,1,5,4)``, ``21-(1,2,6,5)``,
        ``23-(2,3,7,6)`` and ``22-(3,0,4,7)``.
        """,
    )
    BIQUADRATIC_TRIANGLE = _CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_TRIANGLE,
        cell_class=_vtk.vtkBiQuadraticTriangle,
        short_doc="""
        Represent a 2D, 7-node, iso-parametric parabolic triangle.

        The cell includes three mid-edge nodes besides the three triangle vertices
        and a center node.
        """,
        long_doc="""
        The ordering of the three points defining the cell is point ids
        ``(0-2, 3-6)`` where:

        - id ``(3)`` is the mid-edge node between points ``(0,1)``.
        - id ``(4)`` is the mid-edge node between points ``(1,2)``.
        - id ``(5)`` is the mid-edge node between points ``(2,0)``.
        - id ``(6)`` is the center node of the cell.

        """,
    )

    ####################################################################################
    # Cubic, iso-parametric cell
    CUBIC_LINE = _CellTypeTuple(value=_vtk.VTK_CUBIC_LINE, cell_class=_vtk.vtkCubicLine)

    ####################################################################################
    # Special class of cells formed by convex group of points
    CONVEX_POINT_SET = _CellTypeTuple(
        value=_vtk.VTK_CONVEX_POINT_SET,
        cell_class=_vtk.vtkConvexPointSet,
        points_override='variable',
    )

    ####################################################################################
    # Polyhedron cell (consisting of polygonal faces)
    POLYHEDRON = _CellTypeTuple(
        value=_vtk.VTK_POLYHEDRON,
        cell_class=_vtk.vtkPolyhedron,
        points_override='variable',
        edges_override='variable',
        faces_override='variable',
    )

    ####################################################################################
    # Higher order cells in parametric form
    PARAMETRIC_CURVE = _CellTypeTuple(value=_vtk.VTK_PARAMETRIC_CURVE)
    PARAMETRIC_SURFACE = _CellTypeTuple(value=_vtk.VTK_PARAMETRIC_SURFACE)
    PARAMETRIC_TRI_SURFACE = _CellTypeTuple(value=_vtk.VTK_PARAMETRIC_TRI_SURFACE)
    PARAMETRIC_QUAD_SURFACE = _CellTypeTuple(value=_vtk.VTK_PARAMETRIC_QUAD_SURFACE)
    PARAMETRIC_TETRA_REGION = _CellTypeTuple(value=_vtk.VTK_PARAMETRIC_TETRA_REGION)
    PARAMETRIC_HEX_REGION = _CellTypeTuple(value=_vtk.VTK_PARAMETRIC_HEX_REGION)

    ####################################################################################
    # Higher order cells
    HIGHER_ORDER_EDGE = _CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_EDGE)
    HIGHER_ORDER_TRIANGLE = _CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_TRIANGLE)
    HIGHER_ORDER_QUAD = _CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_QUAD)
    HIGHER_ORDER_POLYGON = _CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_POLYGON)
    HIGHER_ORDER_TETRAHEDRON = _CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_TETRAHEDRON)
    HIGHER_ORDER_WEDGE = _CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_WEDGE)
    HIGHER_ORDER_PYRAMID = _CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_PYRAMID)
    HIGHER_ORDER_HEXAHEDRON = _CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_HEXAHEDRON)

    ####################################################################################
    # Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
    LAGRANGE_CURVE = _CellTypeTuple(value=_vtk.VTK_LAGRANGE_CURVE)
    LAGRANGE_TRIANGLE = _CellTypeTuple(value=_vtk.VTK_LAGRANGE_TRIANGLE)
    LAGRANGE_QUADRILATERAL = _CellTypeTuple(value=_vtk.VTK_LAGRANGE_QUADRILATERAL)
    LAGRANGE_TETRAHEDRON = _CellTypeTuple(value=_vtk.VTK_LAGRANGE_TETRAHEDRON)
    LAGRANGE_HEXAHEDRON = _CellTypeTuple(value=_vtk.VTK_LAGRANGE_HEXAHEDRON)
    LAGRANGE_WEDGE = _CellTypeTuple(value=_vtk.VTK_LAGRANGE_WEDGE)
    LAGRANGE_PYRAMID = _CellTypeTuple(value=_vtk.VTK_LAGRANGE_PYRAMID)

    ####################################################################################
    # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    BEZIER_CURVE = _CellTypeTuple(value=_vtk.VTK_BEZIER_CURVE)
    BEZIER_TRIANGLE = _CellTypeTuple(value=_vtk.VTK_BEZIER_TRIANGLE)
    BEZIER_QUADRILATERAL = _CellTypeTuple(value=_vtk.VTK_BEZIER_QUADRILATERAL)
    BEZIER_TETRAHEDRON = _CellTypeTuple(value=_vtk.VTK_BEZIER_TETRAHEDRON)
    BEZIER_HEXAHEDRON = _CellTypeTuple(value=_vtk.VTK_BEZIER_HEXAHEDRON)
    BEZIER_WEDGE = _CellTypeTuple(value=_vtk.VTK_BEZIER_WEDGE)
    BEZIER_PYRAMID = _CellTypeTuple(value=_vtk.VTK_BEZIER_PYRAMID)
