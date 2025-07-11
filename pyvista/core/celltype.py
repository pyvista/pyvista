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

{}

{}

{}
"""

_GRID_TEMPLATE_WITH_IMAGE = """
.. grid:: 1 2 2 2
    :reverse:
    :margin: 1

    .. grid-item::
        :columns: 12 4 4 4

        .. card::
            :class-body: sd-px-0 sd-py-0 sd-rounded-3
            :link: pyvista.examples.cells.{}
            :link-type: any

            .. image:: /../_build/plot_directive/api/examples/_autosummary/pyvista-examples-cells-{}-1_00_00.png

    .. grid-item::
        :columns: 12 8 8 8

{}

{}

{}
"""  # noqa: E501


def _indent_paragraph(string: str, level: int) -> str:
    indentation = ''.join(['    '] * level)
    return textwrap.indent(textwrap.dedent(string).strip(), indentation)


# See link for color names: https://sphinx-design.readthedocs.io/en/latest/badges_buttons.html
_BADGE_COLORS = dict(linear='primary', primary='success', dimension='secondary', geometry='muted')


def _generate_linear_badge(is_linear: bool) -> str:  # noqa: FBT001
    text = 'Linear' if is_linear else 'Non-linear'
    return f':bdg-{_BADGE_COLORS["linear"]}:`{text}`'


def _generate_primary_badge(is_primary: bool) -> str:  # noqa: FBT001
    text = 'Primary' if is_primary else 'Composite'
    return f':bdg-{_BADGE_COLORS["primary"]}:`{text}`'


def _generate_dimension_badge(dimension: int) -> str:
    return f':bdg-{_BADGE_COLORS["dimension"]}:`{dimension}D`'


def _generate_points_badge(num_points: int) -> str:
    return f':bdg-{_BADGE_COLORS["geometry"]}:`Points: {num_points}`'


def _generate_edges_badge(num_edges: int) -> str:
    return f':bdg-{_BADGE_COLORS["geometry"]}:`Edges: {num_edges}`'


def _generate_faces_badge(num_faces: int) -> str:
    return f':bdg-{_BADGE_COLORS["geometry"]}:`Faces: {num_faces}`'


class _CellTypeTuple(NamedTuple):
    value: int
    cell_class: type[_vtk.vtkCell] | None = None
    short_doc: str = ''
    long_doc: str = ''
    example: str | None = None
    points_override: Literal['variable', 'n/a'] | None = None
    edges_override: Literal['variable', 'n/a'] | None = None
    faces_override: Literal['variable', 'n/a'] | None = None


_CELL_TYPE_INFO = dict(
    ####################################################################################
    # Linear cells
    EMPTY_CELL=_CellTypeTuple(
        value=_vtk.VTK_EMPTY_CELL,
        cell_class=_vtk.vtkEmptyCell,
        short_doc="""Used as a place-holder during processing.""",
    ),
    VERTEX=_CellTypeTuple(
        value=_vtk.VTK_VERTEX,
        cell_class=_vtk.vtkVertex,
        example='Vertex',
        short_doc="""
        Represents a point in 3D space.

        The vertex is a primary zero-dimensional cell. It is defined by a single point.
        """,
    ),
    POLY_VERTEX=_CellTypeTuple(
        value=_vtk.VTK_POLY_VERTEX,
        cell_class=_vtk.vtkPolyVertex,
        example='PolyVertex',
        points_override='variable',
        short_doc="""
        Represents a set of points in 3D space.

        The polyvertex is a composite zero-dimensional cell. It is defined by an
        arbitrarily ordered list of points.
        """,
    ),
    LINE=_CellTypeTuple(
        value=_vtk.VTK_LINE,
        cell_class=_vtk.vtkLine,
        example='Line',
        short_doc="""
        Represents a 1D line.

        The line is a primary one-dimensional cell. It is defined by two points.
        The direction along the line is from the first point to the second point.
        """,
    ),
    POLY_LINE=_CellTypeTuple(
        value=_vtk.VTK_POLY_LINE,
        cell_class=_vtk.vtkPolyLine,
        example='PolyLine',
        points_override='variable',
        short_doc="""
        Represents a set of 1D lines.

        The polyline is a composite one-dimensional cell consisting of one or more
        connected lines.
        """,
        long_doc="""
        The polyline is defined by an ordered list of n+1 points, where n is the number
        of lines in the polyline. Each pair of points ``(i, i+1)`` defines a line.
        """,
    ),
    TRIANGLE=_CellTypeTuple(
        value=_vtk.VTK_TRIANGLE,
        cell_class=_vtk.vtkTriangle,
        example='Triangle',
        short_doc="""
        Represents a 2D triangle.

        The triangle is a primary two-dimensional cell. The triangle is defined by a
        counter-clockwise ordered list of three points.
        """,
        long_doc="""
        The order of the points specifies the direction of the surface normal using the
        right-hand rule.
        """,
    ),
    TRIANGLE_STRIP=_CellTypeTuple(
        value=_vtk.VTK_TRIANGLE_STRIP,
        cell_class=_vtk.vtkTriangleStrip,
        example='TriangleStrip',
        points_override='variable',
        edges_override='variable',
        short_doc="""
        Represents a 2D triangle strip.

        The triangle strip is a composite two-dimensional cell consisting of one or more
        triangles. It is a compact representation of triangles connected edge-to-edge.
        """,
        long_doc="""
        The triangle strip is defined by an ordered list of n+2 points, where n is the
        number of triangles. The ordering of the points is such that each set of three
        points ``(i,i+1,i+2)`` with ``0≤i≤n`` defines a triangle.

        The connectivity of a triangle strip is three points defining an
        initial triangle, then for each additional triangle, a single point that,
        combined with the previous two points, defines the next triangle.

        The points defining the triangle strip need not lie in a plane.
        """,
    ),
    POLYGON=_CellTypeTuple(
        value=_vtk.VTK_POLYGON,
        cell_class=_vtk.vtkPolygon,
        example='Polygon',
        points_override='variable',
        edges_override='variable',
        short_doc="""
        Represents a 2D n-sided polygon.

        The polygon is a primary two-dimensional cell. It is defined by an ordered list
        of three or more points lying in a plane.
        """,
        long_doc="""
        The polygon has n edges, where n is the number of points in the polygon.
        Define the polygon with n-points ordered in the counter-clockwise direction.
        Do not repeat the last point.

        The polygon normal is implicitly defined by a counterclockwise ordering of its
        points using the right-hand rule.

        The polygon may be non-convex, but may not have any internal holes, and cannot
        self-intersect.
        """,
    ),
    PIXEL=_CellTypeTuple(
        value=_vtk.VTK_PIXEL,
        cell_class=_vtk.vtkPixel,
        example='Pixel',
        short_doc="""
        Represents a 2D orthogonal quadrilateral.

        The pixel is a primary two-dimensional cell defined by an ordered list of four
        points.

        .. warning::
            This definition of a pixel differs from the conventional definition which
            describes a single constant-valued element in an image. The meaning of this
            term can vary depending on context. See :ref:`image_representations_example`
            for examples.
        """,
        long_doc="""
        The points are ordered in the direction of increasing axis coordinate, starting
        with x, then y, then z.

        Unlike a quadrilateral cell, the corners or a pixel are at right angles and
        aligned along x-y-z coordinate axes. It is used to improve computational
        performance.
        """,
    ),
    QUAD=_CellTypeTuple(
        value=_vtk.VTK_QUAD,
        cell_class=_vtk.vtkQuad,
        example='Quadrilateral',
        short_doc="""
        Represents a 2D quadrilateral.

        The quadrilateral is a primary two-dimensional cell. It is defined by an ordered
        list of four points lying in a plane.
        """,
        long_doc="""
        The four points ``(0,1,2,3)`` are ordered counterclockwise around the
        quadrilateral, defining a surface normal using the right-hand rule.

        The quadrilateral is convex and its edges must not intersect.
        """,
    ),
    TETRA=_CellTypeTuple(
        value=_vtk.VTK_TETRA,
        cell_class=_vtk.vtkTetra,
        example='Tetrahedron',
        short_doc="""
        Represents a 3D tetrahedron.

        The tetrahedron is a primary three-dimensional cell. The tetrahedron is defined
        by a list of four non-planar points. It has six edges and four triangular faces.
        """,
        long_doc="""
        The tetrahedron is defined by the four points ``(0-3)`` where ``(0,1,2)``
        is the base of the tetrahedron which, using the right hand rule, forms a
        triangle whose normal points in the direction of the fourth point.
        """,
    ),
    VOXEL=_CellTypeTuple(
        value=_vtk.VTK_VOXEL,
        cell_class=_vtk.vtkVoxel,
        example='Voxel',
        short_doc="""
        Represents a 3D orthogonal parallelepiped.

        The voxel is a primary three-dimensional cell defined by an ordered list of
        eight points.

        .. warning::
            This definition of a voxel differs from the conventional definition which
            describes a single constant-valued volume element. The meaning of this
            term can vary depending on context. See :ref:`image_representations_example`
            for examples.
        """,
        long_doc="""
        The points are ordered in the direction of increasing coordinate value.

        Unlike a hexahedron cell, a voxel has interior angles of 90 degrees, and its
        sides are parallel to the coordinate axes. It is used to improve computational
        performance.
        """,
    ),
    HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_HEXAHEDRON,
        cell_class=_vtk.vtkHexahedron,
        example='Hexahedron',
        short_doc="""
        Represents a 3D rectangular hexahedron.

        The hexahedron is a primary three-dimensional cell consisting of six
        quadrilateral faces, twelve edges, and eight vertices.
        """,
        long_doc="""
        The hexahedron is defined by the eight points ``(0-7)`` where ``(0,1,2,3)``
        is the base of the hexahedron which, using the right hand rule, forms a
        quadrilateral whose normal points in the direction of the opposite face
        ``(4,5,6,7)``.

        The faces and edges must not intersect any other faces and edges, and the
        hexahedron must be convex.
        """,
    ),
    WEDGE=_CellTypeTuple(
        value=_vtk.VTK_WEDGE,
        cell_class=_vtk.vtkWedge,
        example='Wedge',
        short_doc="""
        Represents a linear 3D wedge.

        The wedge is a primary three-dimensional cell consisting of two triangular
        and three quadrilateral faces.
        """,
        long_doc="""
        The cell is defined by the six points ``(0-5)`` where ``(0,1,2)`` is the
        base of the wedge which, using the right hand rule, forms a triangle whose
        normal points outward (away from the triangular face ``(3,4,5)``).

        The faces and edges must not intersect any other faces and edges, and the wedge
        must be convex.
        """,
    ),
    PYRAMID=_CellTypeTuple(
        value=_vtk.VTK_PYRAMID,
        cell_class=_vtk.vtkPyramid,
        example='Pyramid',
        short_doc="""
        Represents a 3D pyramid.

        The pyramid is a primary three-dimensional cell consisting of a rectangular base
        with four triangular faces. It is defined by an ordered list of five points.
        """,
        long_doc="""
        The pyramid is defined by the five points ``(0-4)`` where ``(0,1,2,3)`` is
        the base of the pyramid which, using the right hand rule, forms a
        quadrilateral whose normal points in the direction of the pyramid apex at
        vertex ``(4)``. The parametric location of vertex ``(4)`` is ``[0, 0, 1]``.

        The four points defining the quadrilateral base plane must be convex.

        The fifth apex point must not be co-planar with the base points.
        """,
    ),
    PENTAGONAL_PRISM=_CellTypeTuple(
        value=_vtk.VTK_PENTAGONAL_PRISM,
        cell_class=_vtk.vtkPentagonalPrism,
        example='PentagonalPrism',
        short_doc="""
        Represents a convex 3D prism with a pentagonal base and five quadrilateral faces.

        The pentagonal prism is a primary three-dimensional cell defined by an ordered
        list of ten points.
        """,
        long_doc="""
        The prism is defined by the ten points ``(0-9)``, where ``(0,1,2,3,4)`` is
        the base of the prism which, using the right hand rule, forms a pentagon
        whose normal points is in the direction of the opposite face ``(5,6,7,8,9)``.

        The faces and edges must not intersect any other faces and edges and the
        pentagon must be convex.
        """,
    ),
    HEXAGONAL_PRISM=_CellTypeTuple(
        value=_vtk.VTK_HEXAGONAL_PRISM,
        cell_class=_vtk.vtkHexagonalPrism,
        example='HexagonalPrism',
        short_doc="""
        Represents a 3D prism with hexagonal base and six quadrilateral faces.

        The hexagonal prism is a primary three-dimensional cell defined by an ordered
        list of twelve points.
        """,
        long_doc="""
        The prism is defined by the twelve points ``(0-11)`` where ``(0,1,2,3,4,5)``
        is the base of the prism which, using the right hand rule, forms a hexagon
        whose normal points is in the direction of the opposite face ``(6,7,8,9,10,11)``.

        The faces and edges must not intersect any other faces and edges and the
        hexagon must be convex.
        """,
    ),
    ####################################################################################
    # Quadratic, isoparametric cells
    QUADRATIC_EDGE=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_EDGE,
        cell_class=_vtk.vtkQuadraticEdge,
        example='QuadraticEdge',
        short_doc="""
        Represents a 1D, 3-node, iso-parametric parabolic line.

        The cell includes a mid-edge node.
        """,
        long_doc="""
        The ordering of the three points defining the cell is point ids
        ``(0,1,2)`` where id ``(2)`` is the mid-edge node.
        """,
    ),
    QUADRATIC_TRIANGLE=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_TRIANGLE,
        cell_class=_vtk.vtkQuadraticTriangle,
        example='QuadraticTriangle',
        short_doc="""
        Represents a 2D, 6-node, iso-parametric parabolic triangle.

        The cell includes a mid-edge node for each of the three edges of the cell.
        """,
        long_doc="""
        The ordering of the six points defining the cell is point ids
        ``(0-2, 3-5)`` where:

        - id ``(3)`` is the mid-edge node between points ``(0,1)``.
        - id ``(4)`` is the mid-edge node between points ``(1,2)``.
        - id ``(5)`` is the mid-edge node between points ``(2,0)``.

        """,
    ),
    QUADRATIC_QUAD=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_QUAD,
        cell_class=_vtk.vtkQuadraticQuad,
        example='QuadraticQuadrilateral',
        short_doc="""
        Represents a 2D, 8-node iso-parametric parabolic quadrilateral element.

        The cell includes a mid-edge node for each of the four edges of the cell.
        """,
        long_doc="""
        The ordering of the eight points defining the cell are point ids
        ``(0-3, 4-7)`` where:

        - ids ``(0-3)`` define the four corner vertices of the quad.
        - ids ``(4-7)`` define the mid-edge nodes ``(0,1)``, ``(1,2)``, ``(2,3)``, ``(3,0)``.

        """,
    ),
    QUADRATIC_POLYGON=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_POLYGON,
        cell_class=_vtk.vtkQuadraticPolygon,
        example='QuadraticPolygon',
        points_override='variable',
        edges_override='variable',
        short_doc="""
        Represents a 2D n-sided (2*n nodes) parabolic polygon.

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
    ),
    QUADRATIC_TETRA=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_TETRA,
        cell_class=_vtk.vtkQuadraticTetra,
        example='QuadraticTetrahedron',
        short_doc="""
        Represents a 3D, 10-node, iso-parametric parabolic tetrahedron.

        The cell includes a mid-edge node on each of the side edges of the tetrahedron.
        """,
        long_doc="""
        The ordering of the ten points defining the cell is point ids ``(0-3, 4-9)``
        where:

        - ids ``(0-3)`` are the four tetra vertices.
        - ids ``(4-9)`` are the mid-edge nodes between ``(0,1)``, ``(1,2)``, ``(2,0)``,
          ``(0,3)``, ``(1,3)``, and ``(2,3)``.

        """,
    ),
    QUADRATIC_HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_HEXAHEDRON,
        cell_class=_vtk.vtkQuadraticHexahedron,
        example='QuadraticHexahedron',
        short_doc="""
        Represents a 3D, 20-node iso-parametric parabolic hexahedron.

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
    ),
    QUADRATIC_WEDGE=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_WEDGE,
        cell_class=_vtk.vtkQuadraticWedge,
        example='QuadraticWedge',
        short_doc="""
        Represents a 3D, 15-node iso-parametric parabolic wedge.

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
    ),
    QUADRATIC_PYRAMID=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_PYRAMID,
        cell_class=_vtk.vtkQuadraticPyramid,
        example='QuadraticPyramid',
        short_doc="""
        Represents a 3D, 13-node iso-parametric parabolic pyramid.

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
    ),
    BIQUADRATIC_QUAD=_CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUAD,
        cell_class=_vtk.vtkBiQuadraticQuad,
        example='BiQuadraticQuadrilateral',
        short_doc="""
        Represents a 2D, 9-node iso-parametric parabolic quadrilateral element with a center-point.

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
    ),
    TRIQUADRATIC_HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_TRIQUADRATIC_HEXAHEDRON,
        cell_class=_vtk.vtkTriQuadraticHexahedron,
        example='TriQuadraticHexahedron',
        short_doc="""
        Represents a 3D, 27-node iso-parametric triquadratic hexahedron.

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
    ),
    QUADRATIC_LINEAR_QUAD=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_LINEAR_QUAD,
        cell_class=_vtk.vtkQuadraticLinearQuad,
        example='QuadraticLinearQuadrilateral',
        short_doc="""
        Represents a 2D, 6-node iso-parametric quadratic-linear quadrilateral element.

        The cell includes a mid-edge node for two of the four edges.
        """,
        long_doc="""
        The ordering of the six points defining the cell are point ids
        ``(0-3, 4-5)`` where:

        - ids ``(0-3)`` define the four corner vertices of the quad.
        - ids ``(4-7)`` define the mid-edge nodes ``(0,1)`` and ``(2,3)``.

        """,
    ),
    QUADRATIC_LINEAR_WEDGE=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_LINEAR_WEDGE,
        cell_class=_vtk.vtkQuadraticLinearWedge,
        example='QuadraticLinearWedge',
        short_doc="""
        Represents a 3D, 12-node iso-parametric linear quadratic wedge.

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
    ),
    BIQUADRATIC_QUADRATIC_WEDGE=_CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUADRATIC_WEDGE,
        cell_class=_vtk.vtkBiQuadraticQuadraticWedge,
        example='BiQuadraticQuadraticWedge',
        short_doc="""
        Represents a 3D, 18-node iso-parametric bi-quadratic wedge.

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
    ),
    BIQUADRATIC_QUADRATIC_HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON,
        cell_class=_vtk.vtkBiQuadraticQuadraticHexahedron,
        example='BiQuadraticQuadraticHexahedron',
        short_doc="""
        Represents a 3D, 24-node iso-parametric biquadratic hexahedron.

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
    ),
    BIQUADRATIC_TRIANGLE=_CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_TRIANGLE,
        cell_class=_vtk.vtkBiQuadraticTriangle,
        example='BiQuadraticTriangle',
        short_doc="""
        Represents a 2D, 7-node, iso-parametric parabolic triangle.

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
    ),
    ####################################################################################
    # Cubic, iso-parametric cell
    CUBIC_LINE=_CellTypeTuple(
        value=_vtk.VTK_CUBIC_LINE,
        cell_class=_vtk.vtkCubicLine,
        example='CubicLine',
        short_doc="""
        Represents a 1D iso-parametric cubic line.

        The cell includes two mid-edge nodes.
        """,
        long_doc="""
        The ordering of the four points defining the cell is point ids ``(0,1,2,3)``
        where id #2 and #3 are the mid-edge nodes.

        The parametric coordinates lie between -1 and 1.
        """,
    ),
    ####################################################################################
    # Special class of cells formed by convex group of points
    CONVEX_POINT_SET=_CellTypeTuple(
        value=_vtk.VTK_CONVEX_POINT_SET,
        cell_class=_vtk.vtkConvexPointSet,
        points_override='variable',
    ),
    ####################################################################################
    # Polyhedron cell (consisting of polygonal faces)
    POLYHEDRON=_CellTypeTuple(
        value=_vtk.VTK_POLYHEDRON,
        cell_class=_vtk.vtkPolyhedron,
        example='Polyhedron',
        short_doc="""
        Represents a 3D cell defined by a set of polygonal faces.

        """,
        long_doc="""
        Polyhedrons must:

        - be watertight: the faces describing the polyhedron should define
          an enclosed volume with a clear “inside” and “outside”
        - have planar faces: all points defining a face should be in the
          same 2D plane
        - not be self-intersecting: for example, a face of the polyhedron
          can't intersect other ones
        - not contain zero-thickness portions: adjacent faces should not
          overlap each other even partially
        - not contain disconnected elements: detached vertice(s), edge(s) or face(s)
        - be simply connected: :vtk:`vtkPolyhedron` must describe a single polyhedron
        - not contain duplicate elements: each point index and each face
          description should be unique
        - not contain “internal” or “external” faces: for each face,
          one side should be “inside” the cell, the other side “outside”

        """,
        points_override='variable',
        edges_override='variable',
        faces_override='variable',
    ),
    ####################################################################################
    # Higher order cells in parametric form
    PARAMETRIC_CURVE=_CellTypeTuple(value=_vtk.VTK_PARAMETRIC_CURVE),
    PARAMETRIC_SURFACE=_CellTypeTuple(value=_vtk.VTK_PARAMETRIC_SURFACE),
    PARAMETRIC_TRI_SURFACE=_CellTypeTuple(value=_vtk.VTK_PARAMETRIC_TRI_SURFACE),
    PARAMETRIC_QUAD_SURFACE=_CellTypeTuple(value=_vtk.VTK_PARAMETRIC_QUAD_SURFACE),
    PARAMETRIC_TETRA_REGION=_CellTypeTuple(value=_vtk.VTK_PARAMETRIC_TETRA_REGION),
    PARAMETRIC_HEX_REGION=_CellTypeTuple(value=_vtk.VTK_PARAMETRIC_HEX_REGION),
    ####################################################################################
    # Higher order cells
    HIGHER_ORDER_EDGE=_CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_EDGE),
    HIGHER_ORDER_TRIANGLE=_CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_TRIANGLE),
    HIGHER_ORDER_QUAD=_CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_QUAD),
    HIGHER_ORDER_POLYGON=_CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_POLYGON),
    HIGHER_ORDER_TETRAHEDRON=_CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_TETRAHEDRON),
    HIGHER_ORDER_WEDGE=_CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_WEDGE),
    HIGHER_ORDER_PYRAMID=_CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_PYRAMID),
    HIGHER_ORDER_HEXAHEDRON=_CellTypeTuple(value=_vtk.VTK_HIGHER_ORDER_HEXAHEDRON),
    ####################################################################################
    # Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
    LAGRANGE_CURVE=_CellTypeTuple(value=_vtk.VTK_LAGRANGE_CURVE),
    LAGRANGE_TRIANGLE=_CellTypeTuple(value=_vtk.VTK_LAGRANGE_TRIANGLE),
    LAGRANGE_QUADRILATERAL=_CellTypeTuple(value=_vtk.VTK_LAGRANGE_QUADRILATERAL),
    LAGRANGE_TETRAHEDRON=_CellTypeTuple(value=_vtk.VTK_LAGRANGE_TETRAHEDRON),
    LAGRANGE_HEXAHEDRON=_CellTypeTuple(value=_vtk.VTK_LAGRANGE_HEXAHEDRON),
    LAGRANGE_WEDGE=_CellTypeTuple(value=_vtk.VTK_LAGRANGE_WEDGE),
    LAGRANGE_PYRAMID=_CellTypeTuple(value=_vtk.VTK_LAGRANGE_PYRAMID),
    ####################################################################################
    # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    BEZIER_CURVE=_CellTypeTuple(value=_vtk.VTK_BEZIER_CURVE),
    BEZIER_TRIANGLE=_CellTypeTuple(value=_vtk.VTK_BEZIER_TRIANGLE),
    BEZIER_QUADRILATERAL=_CellTypeTuple(value=_vtk.VTK_BEZIER_QUADRILATERAL),
    BEZIER_TETRAHEDRON=_CellTypeTuple(value=_vtk.VTK_BEZIER_TETRAHEDRON),
    BEZIER_HEXAHEDRON=_CellTypeTuple(value=_vtk.VTK_BEZIER_HEXAHEDRON),
    BEZIER_WEDGE=_CellTypeTuple(value=_vtk.VTK_BEZIER_WEDGE),
    BEZIER_PYRAMID=_CellTypeTuple(value=_vtk.VTK_BEZIER_PYRAMID),
)
if hasattr(_vtk, 'VTK_TRIQUADRATIC_PYRAMID'):
    _CELL_TYPE_INFO['TRIQUADRATIC_PYRAMID'] = _CellTypeTuple(
        value=_vtk.VTK_TRIQUADRATIC_PYRAMID,
        cell_class=_vtk.vtkTriQuadraticPyramid,
        example='TriQuadraticPyramid',
        short_doc="""
        Represents a second order 3D iso-parametric 19-node pyramid.

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


class CellType(IntEnum):
    """Define types of cells.

    Cells are defined by specifying a type in combination with an ordered list of points.
    The ordered list, often referred to as the connectivity list, combined with the
    type specification, implicitly defines the topology of the cell. The x-y-z point
    coordinates define the cell geometry.

    Although point coordinates are defined in three dimensions, the cell topology can
    be 0, 1, 2, or 3-dimensional.

    Cells can be primary (e.g. triangle) or composite (e.g. triangle strip). Composite
    cells consist of one or more primary cells, while primary cells cannot be
    decomposed.

    Cells can also be characterized as linear or non-linear. Linear cells use
    linear or constant interpolation. Non-linear cells may use quadratic,
    cubic, or some other interpolation.

    This enumeration defines all cell types used in VTK and supported by PyVista. The
    type(s) of cell(s) to use is typically chosen based on application need, such as
    graphics rendering or numerical simulation.

    .. seealso::

        `VTK Book: Cell Types <https://book.vtk.org/en/latest/VTKBook/05Chapter5.html#cell-types>`_
            VTK reference about cell types.

        `vtkCellType.h <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_
            List of all cell types defined in VTK.

        :ref:`linear_cells_example`
            Detailed example using linear cells.

        :ref:`create_polyhedron_example`
            Example creating a mesh with :attr:`~pyvista.CellType.POLYHEDRON` cells.

        :ref:`create_polydata_strips_example`
            Example creating a mesh with :attr:`~pyvista.CellType.TRIANGLE_STRIP` cells.

        :mod:`pyvista.examples.cells`
            Examples creating a mesh comprising a single cell.

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

    def __new__(  # noqa: PYI034
        cls: type[CellType],
        value: int,
        _cell_class: type[_vtk.vtkCell] | None = None,
        _short_doc: str = '',
        _long_doc: str = '',
        _example: str | None = None,
        _points_override: Literal['variable', 'n/a'] | None = None,
        _edges_override: Literal['variable', 'n/a'] | None = None,
        _faces_override: Literal['variable', 'n/a'] | None = None,
    ) -> CellType:
        """Create new enum.

        Optionally specify documentation info.

        .. note::

            When specifying multi-line ``short_doc`` or ``long_doc`` strings, the
            lines *must* be all aligned. I.e. do not put the first line immediately
            after the triple quotes; instead put the first line of text on a new line.

        Parameters
        ----------
        value : int
            Integer value of the cell type.

        _cell_class : type[:vtk:`vtkCell`], optional
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

        _short_doc = textwrap.dedent(_short_doc).strip()
        _long_doc = textwrap.dedent(_long_doc).strip()

        # Generate cell type documentation if specified
        if _cell_class or _short_doc or _long_doc or _example:
            badges = ''
            if _cell_class:
                cell = _cell_class()
                linear_badge = _generate_linear_badge(cell.IsLinear())  # type: ignore[arg-type]
                primary_badge = _generate_primary_badge(cell.IsPrimaryCell())  # type: ignore[arg-type]
                dimension_badge = _generate_dimension_badge(cell.GetCellDimension())

                points = _points_override or cell.GetNumberOfPoints()
                points_badge = _generate_points_badge(points)  # type: ignore[arg-type]

                edges = _edges_override or cell.GetNumberOfEdges()
                edges_badge = _generate_edges_badge(edges)  # type: ignore[arg-type]

                faces = _faces_override or cell.GetNumberOfFaces()
                faces_badge = _generate_faces_badge(faces)  # type: ignore[arg-type]

                badges = _indent_paragraph(
                    f'{linear_badge} {primary_badge} {dimension_badge}\n'
                    f'{points_badge} {edges_badge} {faces_badge}',
                    level=2,
                )

                # Add additional references to VTK docs
                cell_class_ref = f':vtk:`{_cell_class.__name__}`'
                see_also = f'See also {cell_class_ref}.'
                _long_doc += f'\n\n{see_also}'

            _short_doc = _indent_paragraph(_short_doc, level=2)
            _long_doc = _indent_paragraph(
                _DROPDOWN_TEMPLATE.format(_indent_paragraph(_long_doc, level=1)), level=2
            )

            self.__doc__ += (
                _GRID_TEMPLATE_NO_IMAGE.format(badges, _short_doc, _long_doc)
                if _example is None
                else _GRID_TEMPLATE_WITH_IMAGE.format(
                    _example, _example, badges, _short_doc, _long_doc
                )
            )

        return self

    EMPTY_CELL = _CELL_TYPE_INFO['EMPTY_CELL']
    VERTEX = _CELL_TYPE_INFO['VERTEX']
    POLY_VERTEX = _CELL_TYPE_INFO['POLY_VERTEX']
    LINE = _CELL_TYPE_INFO['LINE']
    POLY_LINE = _CELL_TYPE_INFO['POLY_LINE']
    TRIANGLE = _CELL_TYPE_INFO['TRIANGLE']
    TRIANGLE_STRIP = _CELL_TYPE_INFO['TRIANGLE_STRIP']
    POLYGON = _CELL_TYPE_INFO['POLYGON']
    PIXEL = _CELL_TYPE_INFO['PIXEL']
    QUAD = _CELL_TYPE_INFO['QUAD']
    TETRA = _CELL_TYPE_INFO['TETRA']
    VOXEL = _CELL_TYPE_INFO['VOXEL']
    HEXAHEDRON = _CELL_TYPE_INFO['HEXAHEDRON']
    WEDGE = _CELL_TYPE_INFO['WEDGE']
    PYRAMID = _CELL_TYPE_INFO['PYRAMID']
    PENTAGONAL_PRISM = _CELL_TYPE_INFO['PENTAGONAL_PRISM']
    HEXAGONAL_PRISM = _CELL_TYPE_INFO['HEXAGONAL_PRISM']
    QUADRATIC_EDGE = _CELL_TYPE_INFO['QUADRATIC_EDGE']
    QUADRATIC_TRIANGLE = _CELL_TYPE_INFO['QUADRATIC_TRIANGLE']
    QUADRATIC_QUAD = _CELL_TYPE_INFO['QUADRATIC_QUAD']
    QUADRATIC_POLYGON = _CELL_TYPE_INFO['QUADRATIC_POLYGON']
    QUADRATIC_TETRA = _CELL_TYPE_INFO['QUADRATIC_TETRA']
    QUADRATIC_HEXAHEDRON = _CELL_TYPE_INFO['QUADRATIC_HEXAHEDRON']
    QUADRATIC_WEDGE = _CELL_TYPE_INFO['QUADRATIC_WEDGE']
    QUADRATIC_PYRAMID = _CELL_TYPE_INFO['QUADRATIC_PYRAMID']
    BIQUADRATIC_QUAD = _CELL_TYPE_INFO['BIQUADRATIC_QUAD']
    TRIQUADRATIC_HEXAHEDRON = _CELL_TYPE_INFO['TRIQUADRATIC_HEXAHEDRON']
    if hasattr(_vtk, 'VTK_TRIQUADRATIC_PYRAMID'):
        TRIQUADRATIC_PYRAMID = _CELL_TYPE_INFO['TRIQUADRATIC_PYRAMID']
    QUADRATIC_LINEAR_QUAD = _CELL_TYPE_INFO['QUADRATIC_LINEAR_QUAD']
    QUADRATIC_LINEAR_WEDGE = _CELL_TYPE_INFO['QUADRATIC_LINEAR_WEDGE']
    BIQUADRATIC_QUADRATIC_WEDGE = _CELL_TYPE_INFO['BIQUADRATIC_QUADRATIC_WEDGE']
    BIQUADRATIC_QUADRATIC_HEXAHEDRON = _CELL_TYPE_INFO['BIQUADRATIC_QUADRATIC_HEXAHEDRON']
    BIQUADRATIC_TRIANGLE = _CELL_TYPE_INFO['BIQUADRATIC_TRIANGLE']
    CUBIC_LINE = _CELL_TYPE_INFO['CUBIC_LINE']
    CONVEX_POINT_SET = _CELL_TYPE_INFO['CONVEX_POINT_SET']
    POLYHEDRON = _CELL_TYPE_INFO['POLYHEDRON']
    PARAMETRIC_CURVE = _CELL_TYPE_INFO['PARAMETRIC_CURVE']
    PARAMETRIC_SURFACE = _CELL_TYPE_INFO['PARAMETRIC_SURFACE']
    PARAMETRIC_TRI_SURFACE = _CELL_TYPE_INFO['PARAMETRIC_TRI_SURFACE']
    PARAMETRIC_QUAD_SURFACE = _CELL_TYPE_INFO['PARAMETRIC_QUAD_SURFACE']
    PARAMETRIC_TETRA_REGION = _CELL_TYPE_INFO['PARAMETRIC_TETRA_REGION']
    PARAMETRIC_HEX_REGION = _CELL_TYPE_INFO['PARAMETRIC_HEX_REGION']
    HIGHER_ORDER_EDGE = _CELL_TYPE_INFO['HIGHER_ORDER_EDGE']
    HIGHER_ORDER_TRIANGLE = _CELL_TYPE_INFO['HIGHER_ORDER_TRIANGLE']
    HIGHER_ORDER_QUAD = _CELL_TYPE_INFO['HIGHER_ORDER_QUAD']
    HIGHER_ORDER_POLYGON = _CELL_TYPE_INFO['HIGHER_ORDER_POLYGON']
    HIGHER_ORDER_TETRAHEDRON = _CELL_TYPE_INFO['HIGHER_ORDER_TETRAHEDRON']
    HIGHER_ORDER_WEDGE = _CELL_TYPE_INFO['HIGHER_ORDER_WEDGE']
    HIGHER_ORDER_PYRAMID = _CELL_TYPE_INFO['HIGHER_ORDER_PYRAMID']
    HIGHER_ORDER_HEXAHEDRON = _CELL_TYPE_INFO['HIGHER_ORDER_HEXAHEDRON']
    LAGRANGE_CURVE = _CELL_TYPE_INFO['LAGRANGE_CURVE']
    LAGRANGE_TRIANGLE = _CELL_TYPE_INFO['LAGRANGE_TRIANGLE']
    LAGRANGE_QUADRILATERAL = _CELL_TYPE_INFO['LAGRANGE_QUADRILATERAL']
    LAGRANGE_TETRAHEDRON = _CELL_TYPE_INFO['LAGRANGE_TETRAHEDRON']
    LAGRANGE_HEXAHEDRON = _CELL_TYPE_INFO['LAGRANGE_HEXAHEDRON']
    LAGRANGE_WEDGE = _CELL_TYPE_INFO['LAGRANGE_WEDGE']
    LAGRANGE_PYRAMID = _CELL_TYPE_INFO['LAGRANGE_PYRAMID']
    BEZIER_CURVE = _CELL_TYPE_INFO['BEZIER_CURVE']
    BEZIER_TRIANGLE = _CELL_TYPE_INFO['BEZIER_TRIANGLE']
    BEZIER_QUADRILATERAL = _CELL_TYPE_INFO['BEZIER_QUADRILATERAL']
    BEZIER_TETRAHEDRON = _CELL_TYPE_INFO['BEZIER_TETRAHEDRON']
    BEZIER_HEXAHEDRON = _CELL_TYPE_INFO['BEZIER_HEXAHEDRON']
    BEZIER_WEDGE = _CELL_TYPE_INFO['BEZIER_WEDGE']
    BEZIER_PYRAMID = _CELL_TYPE_INFO['BEZIER_PYRAMID']
