"""Define types of cells."""

from __future__ import annotations

from enum import IntEnum
import textwrap
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import cast
from typing import get_args

from pyvista.core import _vtk_core as _vtk
from pyvista.core._vtk_utilities import vtk_version_info


class _classproperty(property):  # noqa: N801
    """Read-only class property decorator.

    Use this decaorator as an alternative to chaining `@classmethod`
    and `@property` which is deprecated.

    See:
    - https://docs.python.org/library/functions.html#classmethod
    - https://stackoverflow.com/a/13624858

    Examples
    --------
    >>> from pyvista.core.utilities.misc import _classproperty
    >>> class Foo:
    ...     @_classproperty
    ...     def bar(cls): ...

    """

    def __get__(self: property, owner_self: Any, owner_cls: type | None = None) -> Any:
        return self.fget(owner_cls)  # type: ignore[misc]


_Dimension = Literal[0, 1, 2, 3]
_DIMENSION_MAP: dict[_Dimension, frozenset[CellType]] = {}
PLACEHOLDER = 'IMAGE-HASH-PLACEHOLDER'

_GRID_TEMPLATE_NO_IMAGE = """
.. grid:: 1
    :margin: 1

    .. grid-item::

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

            .. image:: /../_build/plot_directive/api/examples/_autosummary/pyvista-examples-cells-{}-{}_00_00.png

    .. grid-item::
        :columns: 12 8 8 8

{}

{}
"""  # noqa: E501


def _indent_paragraph(string: str, level: int) -> str:
    indentation = ''.join(['    '] * level)
    return textwrap.indent(textwrap.dedent(string).strip(), indentation)


# See link for color names: https://sphinx-design.readthedocs.io/en/latest/badges_buttons.html
_BADGE_COLORS = dict(
    linear='primary', composite='success', dimension='secondary', geometry='muted'
)


def _generate_badge(*, kind: str, text: str, attr: str) -> str:
    color = _BADGE_COLORS[kind]
    ref = f'pyvista.CellType.{attr}'
    return f':bdg-link-{color}:`{text} <{ref}>`'


def _generate_linear_badge(is_linear: bool) -> str:  # noqa: FBT001
    text = 'Linear' if is_linear else 'Non-linear'
    return _generate_badge(kind='linear', text=text, attr='is_linear')


def _generate_composite_badge(is_composite: bool) -> str:  # noqa: FBT001
    text = 'Composite' if is_composite else 'Primary'
    return _generate_badge(kind='composite', text=text, attr='is_composite')


def _generate_dimension_badge(dimension: int) -> str:
    return _generate_badge(kind='dimension', text=f'{dimension}D', attr='dimension')


def _generate_points_badge(n_points: int) -> str:
    return _generate_badge(kind='geometry', text=f'Points: {n_points}', attr='n_points')


def _generate_edges_badge(n_edges: int) -> str:
    return _generate_badge(kind='geometry', text=f'Edges: {n_edges}', attr='n_edges')


def _generate_faces_badge(n_faces: int) -> str:
    return _generate_badge(kind='geometry', text=f'Faces: {n_faces}', attr='n_faces')


class _CellTypeTuple(NamedTuple):
    value: int
    doc: str = ''
    example: str | None = None
    variable_points: bool = False
    variable_edges: bool = False
    variable_faces: bool = False


_CELL_TYPE_INFO = dict(
    ####################################################################################
    # Linear cells
    EMPTY_CELL=_CellTypeTuple(
        value=_vtk.VTK_EMPTY_CELL,
        example='Empty',
        doc="""Used as a place-holder during processing.""",
    ),
    VERTEX=_CellTypeTuple(
        value=_vtk.VTK_VERTEX,
        example='Vertex',
        doc="""
        Represents a point in 3D space.

        The vertex is a primary zero-dimensional cell. It is defined by a single point.
        """,
    ),
    POLY_VERTEX=_CellTypeTuple(
        value=_vtk.VTK_POLY_VERTEX,
        example='PolyVertex',
        variable_points=True,
        doc="""
        Represents a set of points in 3D space.

        The polyvertex is a composite zero-dimensional cell. It is defined by an
        arbitrarily ordered list of points.
        """,
    ),
    LINE=_CellTypeTuple(
        value=_vtk.VTK_LINE,
        example='Line',
        doc="""
        Represents a 1D line.

        The line is a primary one-dimensional cell. It is defined by two points.
        The direction along the line is from the first point to the second point.
        """,
    ),
    POLY_LINE=_CellTypeTuple(
        value=_vtk.VTK_POLY_LINE,
        example='PolyLine',
        variable_points=True,
        doc="""
        Represents a set of 1D lines.

        The polyline is a composite one-dimensional cell consisting of one or more
        connected lines.
        """,
    ),
    TRIANGLE=_CellTypeTuple(
        value=_vtk.VTK_TRIANGLE,
        example='Triangle',
        doc="""
        Represents a 2D triangle.

        The triangle is a primary two-dimensional cell. The triangle is defined by a
        counter-clockwise ordered list of three points.
        """,
    ),
    TRIANGLE_STRIP=_CellTypeTuple(
        value=_vtk.VTK_TRIANGLE_STRIP,
        example='TriangleStrip',
        variable_points=True,
        variable_edges=True,
        doc="""
        Represents a 2D triangle strip.

        The triangle strip is a composite two-dimensional cell consisting of one or more
        triangles. It is a compact representation of triangles connected edge-to-edge.
        """,
    ),
    POLYGON=_CellTypeTuple(
        value=_vtk.VTK_POLYGON,
        example='Polygon',
        variable_points=True,
        variable_edges=True,
        doc="""
        Represents a 2D n-sided polygon.

        The polygon is a primary two-dimensional cell. It is defined by an ordered list
        of three or more points lying in a plane.
        """,
    ),
    PIXEL=_CellTypeTuple(
        value=_vtk.VTK_PIXEL,
        example='Pixel',
        doc="""
        Represents a 2D orthogonal quadrilateral.

        The pixel is a primary two-dimensional cell defined by an ordered list of four
        points.

        .. warning::
            This definition of a pixel differs from the conventional definition which
            describes a single constant-valued element in an image. The meaning of this
            term can vary depending on context. See :ref:`image_representations_example`
            for examples.
        """,
    ),
    QUAD=_CellTypeTuple(
        value=_vtk.VTK_QUAD,
        example='Quadrilateral',
        doc="""
        Represents a 2D quadrilateral.

        The quadrilateral is a primary two-dimensional cell. It is defined by an ordered
        list of four points lying in a plane.
        """,
    ),
    TETRA=_CellTypeTuple(
        value=_vtk.VTK_TETRA,
        example='Tetrahedron',
        doc="""
        Represents a 3D tetrahedron.

        The tetrahedron is a primary three-dimensional cell. The tetrahedron is defined
        by a list of four non-planar points. It has six edges and four triangular faces.
        """,
    ),
    VOXEL=_CellTypeTuple(
        value=_vtk.VTK_VOXEL,
        example='Voxel',
        doc="""
        Represents a 3D orthogonal parallelepiped.

        The voxel is a primary three-dimensional cell defined by an ordered list of
        eight points.

        .. warning::
            This definition of a voxel differs from the conventional definition which
            describes a single constant-valued volume element. The meaning of this
            term can vary depending on context. See :ref:`image_representations_example`
            for examples.
        """,
    ),
    HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_HEXAHEDRON,
        example='Hexahedron',
        doc="""
        Represents a 3D rectangular hexahedron.

        The hexahedron is a primary three-dimensional cell consisting of six
        quadrilateral faces, twelve edges, and eight vertices.
        """,
    ),
    WEDGE=_CellTypeTuple(
        value=_vtk.VTK_WEDGE,
        example='Wedge',
        doc="""
        Represents a linear 3D wedge.

        The wedge is a primary three-dimensional cell consisting of two triangular
        and three quadrilateral faces.
        """,
    ),
    PYRAMID=_CellTypeTuple(
        value=_vtk.VTK_PYRAMID,
        example='Pyramid',
        doc="""
        Represents a 3D pyramid.

        The pyramid is a primary three-dimensional cell consisting of a rectangular base
        with four triangular faces. It is defined by an ordered list of five points.
        """,
    ),
    PENTAGONAL_PRISM=_CellTypeTuple(
        value=_vtk.VTK_PENTAGONAL_PRISM,
        example='PentagonalPrism',
        doc="""
        Represents a convex 3D prism with a pentagonal base and five quadrilateral faces.

        The pentagonal prism is a primary three-dimensional cell defined by an ordered
        list of ten points.
        """,
    ),
    HEXAGONAL_PRISM=_CellTypeTuple(
        value=_vtk.VTK_HEXAGONAL_PRISM,
        example='HexagonalPrism',
        doc="""
        Represents a 3D prism with hexagonal base and six quadrilateral faces.

        The hexagonal prism is a primary three-dimensional cell defined by an ordered
        list of twelve points.
        """,
    ),
    ####################################################################################
    # Quadratic, isoparametric cells
    QUADRATIC_EDGE=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_EDGE,
        example='QuadraticEdge',
        doc="""
        Represents a 1D, 3-node, iso-parametric parabolic line.

        The cell includes a mid-edge node.
        """,
    ),
    QUADRATIC_TRIANGLE=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_TRIANGLE,
        example='QuadraticTriangle',
        doc="""
        Represents a 2D, 6-node, iso-parametric parabolic triangle.

        The cell includes a mid-edge node for each of the three edges of the cell.
        """,
    ),
    QUADRATIC_QUAD=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_QUAD,
        example='QuadraticQuadrilateral',
        doc="""
        Represents a 2D, 8-node iso-parametric parabolic quadrilateral element.

        The cell includes a mid-edge node for each of the four edges of the cell.
        """,
    ),
    QUADRATIC_POLYGON=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_POLYGON,
        example='QuadraticPolygon',
        variable_points=True,
        variable_edges=True,
        doc="""
        Represents a 2D n-sided (2*n nodes) parabolic polygon.

        The polygon cannot have any internal holes, and cannot self-intersect.
        The cell includes a mid-edge node for each of the n edges of the cell.
        """,
    ),
    QUADRATIC_TETRA=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_TETRA,
        example='QuadraticTetrahedron',
        doc="""
        Represents a 3D, 10-node, iso-parametric parabolic tetrahedron.

        The cell includes a mid-edge node on each of the side edges of the tetrahedron.
        """,
    ),
    QUADRATIC_HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_HEXAHEDRON,
        example='QuadraticHexahedron',
        doc="""
        Represents a 3D, 20-node iso-parametric parabolic hexahedron.

        The cell includes a mid-edge node.
        """,
    ),
    QUADRATIC_WEDGE=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_WEDGE,
        example='QuadraticWedge',
        doc="""
        Represents a 3D, 15-node iso-parametric parabolic wedge.

        The cell includes a mid-edge node.
        """,
    ),
    QUADRATIC_PYRAMID=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_PYRAMID,
        example='QuadraticPyramid',
        doc="""
        Represents a 3D, 13-node iso-parametric parabolic pyramid.

        The cell includes a mid-edge node.
        """,
    ),
    BIQUADRATIC_QUAD=_CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUAD,
        example='BiQuadraticQuadrilateral',
        doc="""
        Represents a 2D, 9-node iso-parametric parabolic quadrilateral element with a center-point.

        The cell includes a mid-edge node for each of the four edges of the cell and
        a center node at the surface.
        """,
    ),
    TRIQUADRATIC_HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_TRIQUADRATIC_HEXAHEDRON,
        example='TriQuadraticHexahedron',
        doc="""
        Represents a 3D, 27-node iso-parametric triquadratic hexahedron.

        The cell includes 8 edge nodes, 12 mid-edge nodes, 6 mid-face nodes and one
        mid-volume node.
        """,
    ),
    TRIQUADRATIC_PYRAMID=_CellTypeTuple(
        value=_vtk.VTK_TRIQUADRATIC_PYRAMID,
        example='TriQuadraticPyramid',
        doc="""
        Represents a second order 3D iso-parametric 19-node pyramid.

        The cell includes 5 corner nodes, 8 mid-edge nodes, 5 mid-face nodes,
        and 1 volumetric centroid node.
        """,
    ),
    QUADRATIC_LINEAR_QUAD=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_LINEAR_QUAD,
        example='QuadraticLinearQuadrilateral',
        doc="""
        Represents a 2D, 6-node iso-parametric quadratic-linear quadrilateral element.

        The cell includes a mid-edge node for two of the four edges.
        """,
    ),
    QUADRATIC_LINEAR_WEDGE=_CellTypeTuple(
        value=_vtk.VTK_QUADRATIC_LINEAR_WEDGE,
        example='QuadraticLinearWedge',
        doc="""
        Represents a 3D, 12-node iso-parametric linear quadratic wedge.

        The cell includes mid-edge node in the triangle edges.
        """,
    ),
    BIQUADRATIC_QUADRATIC_WEDGE=_CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUADRATIC_WEDGE,
        example='BiQuadraticQuadraticWedge',
        doc="""
        Represents a 3D, 18-node iso-parametric bi-quadratic wedge.

        The cell includes a mid-edge node.
        """,
    ),
    BIQUADRATIC_QUADRATIC_HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON,
        example='BiQuadraticQuadraticHexahedron',
        doc="""
        Represents a 3D, 24-node iso-parametric biquadratic hexahedron.

        The cell includes mid-edge and center-face nodes.
        """,
    ),
    BIQUADRATIC_TRIANGLE=_CellTypeTuple(
        value=_vtk.VTK_BIQUADRATIC_TRIANGLE,
        example='BiQuadraticTriangle',
        doc="""
        Represents a 2D, 7-node, iso-parametric parabolic triangle.

        The cell includes three mid-edge nodes besides the three triangle vertices
        and a center node.
        """,
    ),
    ####################################################################################
    # Cubic, iso-parametric cell
    CUBIC_LINE=_CellTypeTuple(
        value=_vtk.VTK_CUBIC_LINE,
        example='CubicLine',
        doc="""
        Represents a 1D iso-parametric cubic line.

        The cell includes two mid-edge nodes.
        """,
    ),
    ####################################################################################
    # Special class of cells formed by convex group of points
    CONVEX_POINT_SET=_CellTypeTuple(
        value=_vtk.VTK_CONVEX_POINT_SET,
        example='ConvexPointSet',
        doc="""
        Represents a 3D cell defined by a convex set of points.
        """,
        variable_points=True,
        variable_edges=True,
        variable_faces=True,
    ),
    ####################################################################################
    # Polyhedron cell (consisting of polygonal faces)
    POLYHEDRON=_CellTypeTuple(
        value=_vtk.VTK_POLYHEDRON,
        example='Polyhedron',
        doc="""
        Represents a 3D cell defined by a set of polygonal faces.

        """,
        variable_points=True,
        variable_edges=True,
        variable_faces=True,
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
    LAGRANGE_CURVE=_CellTypeTuple(
        value=_vtk.VTK_LAGRANGE_CURVE,
        example='LagrangeCurve',
        variable_points=True,
        doc="""
        Lagrange representation of a curve with arbitrary order.
        """,
    ),
    LAGRANGE_TRIANGLE=_CellTypeTuple(
        value=_vtk.VTK_LAGRANGE_TRIANGLE,
        example='LagrangeTriangle',
        variable_points=True,
        doc="""
        Lagrange representation of a triangle with arbitrary order.
        """,
    ),
    LAGRANGE_QUADRILATERAL=_CellTypeTuple(
        value=_vtk.VTK_LAGRANGE_QUADRILATERAL,
        example='LagrangeQuadrilateral',
        variable_points=True,
        doc="""
        Lagrange representation of a quadrilateral with arbitrary order.
        """,
    ),
    LAGRANGE_TETRAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_LAGRANGE_TETRAHEDRON,
        example='LagrangeTetrahedron',
        variable_points=True,
        doc="""
        Lagrange representation of a tetrahedron with arbitrary order.
        """,
    ),
    LAGRANGE_HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_LAGRANGE_HEXAHEDRON,
        example='LagrangeHexahedron',
        variable_points=True,
        doc="""
        Lagrange representation of a hexahedron with arbitrary order.
        """,
    ),
    LAGRANGE_WEDGE=_CellTypeTuple(
        value=_vtk.VTK_LAGRANGE_WEDGE,
        example='LagrangeWedge',
        variable_points=True,
        doc="""
        Lagrange representation of a wedge with arbitrary order.
        """,
    ),
    LAGRANGE_PYRAMID=_CellTypeTuple(
        value=_vtk.VTK_LAGRANGE_PYRAMID,
        doc="""
        Lagrange representation of a pyramid with arbitrary order.
        """,
    ),
    ####################################################################################
    # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    BEZIER_CURVE=_CellTypeTuple(
        value=_vtk.VTK_BEZIER_CURVE,
        example='BezierCurve',
        variable_points=True,
        doc="""
        Bezier representation of a curve with arbitrary order.
        """,
    ),
    BEZIER_TRIANGLE=_CellTypeTuple(
        value=_vtk.VTK_BEZIER_TRIANGLE,
        example='BezierTriangle',
        variable_points=True,
        doc="""
        Bezier representation of a triangle with arbitrary order.
        """,
    ),
    BEZIER_QUADRILATERAL=_CellTypeTuple(
        value=_vtk.VTK_BEZIER_QUADRILATERAL,
        example='BezierQuadrilateral',
        variable_points=True,
        doc="""
        Bezier representation of a quadrilateral with arbitrary order.
        """,
    ),
    BEZIER_TETRAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_BEZIER_TETRAHEDRON,
        example='BezierTetrahedron',
        variable_points=True,
        doc="""
        Bezier representation of a tetrahedron with arbitrary order.
        """,
    ),
    BEZIER_HEXAHEDRON=_CellTypeTuple(
        value=_vtk.VTK_BEZIER_HEXAHEDRON,
        example='BezierHexahedron',
        variable_points=True,
        doc="""
        Bezier representation of a hexahedron with arbitrary order.
        """,
    ),
    BEZIER_WEDGE=_CellTypeTuple(
        value=_vtk.VTK_BEZIER_WEDGE,
        example='BezierWedge',
        variable_points=True,
        doc="""
        Bezier representation of a wedge with arbitrary order.
        """,
    ),
    BEZIER_PYRAMID=_CellTypeTuple(
        value=_vtk.VTK_BEZIER_PYRAMID,
        doc="""
        Bezier representation of a pyramid with arbitrary order.
        """,
    ),
)


class CellType(IntEnum):
    """Define types of cells.

    Cells are defined by specifying a type in combination with an ordered list of points.
    The ordered list, often referred to as the connectivity list, combined with the
    type specification, implicitly defines the topology of the cell. The x-y-z point
    coordinates define the cell geometry.

    Although point coordinates are defined in three dimensions, the cell topology can
    be 0, 1, 2, or 3-dimensional. Use :attr:`dimension` to check the topological dimension.

    Cells can be primary (e.g. triangle) or composite (e.g. triangle strip). Composite
    cells consist of one or more primary cells, while primary cells cannot be
    decomposed. Use :attr:`is_composite` to check if a cell type is primary or composite.

    Cells can also be characterized as linear or non-linear. Linear cells use
    linear or constant interpolation, while non-linear cells may use quadratic,
    cubic, or some other interpolation. Use :attr:`is_linear` to check if a cell type is linear.

    Higher order interpolation is possible using Lagrange or Bézier cells. For Lagrange cells,
    all the points lie on the interpolation curve, whereas for the Bézier cells, only the points at
    the extremities are interpolatory.

    This enumeration defines all cell types used in VTK and supported by PyVista. The
    type(s) of cell(s) to use is typically chosen based on application need, such as
    graphics rendering or numerical simulation.

    .. seealso::

        :class:`pyvista.Cell`
            Wrapper for instances of :vtk:`vtkCell`.

        :mod:`pyvista.examples.cells`
            Examples creating a mesh comprising a single cell.

        :ref:`linear_cells_example`
            Detailed example using linear cells.

        :ref:`create_polyhedron_example`
            Example creating a mesh with :attr:`~pyvista.CellType.POLYHEDRON` cells.

        :ref:`create_polydata_strips_example`
            Example creating a mesh with :attr:`~pyvista.CellType.TRIANGLE_STRIP` cells.

    References
    ----------
    `VTK Book: Cell Types <https://book.vtk.org/en/latest/VTKBook/05Chapter5.html#cell-types>`_
        VTK reference about cell types.

    `Modeling Lagrange Finite Elements in VTK <https://www.kitware.com//modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/>`_
        VTK blog post about Lagrange cells.

    `Implementation of rational Bézier cells into VTK <https://www.kitware.com/implementation-of-rational-bezier-cells-into-vtk/>`_
        VTK blog post about Bezier cells.

    `vtkCellType.h <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_
        List of all cell types defined in VTK.

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

    # Attributes populated in __new__
    _vtk_class: type[_vtk.vtkCell] | None
    _dimension: _Dimension
    _is_linear: bool
    _is_composite: bool
    _n_points: int
    _n_edges: int
    _n_faces: int
    _example: str | None

    def __new__(  # noqa: PYI034
        cls: type[CellType],
        value: int,
        _doc: str = '',
        _example: str | None = None,
        _variable_points: bool = False,  # noqa: FBT001, FBT002
        _variable_edges: bool = False,  # noqa: FBT001, FBT002
        _variable_faces: bool = False,  # noqa: FBT001, FBT002
    ) -> CellType:
        """Create new enum.

        Optionally specify documentation info.

        .. note::

            When specifying multi-line ``doc`` strings, the lines *must* be all aligned.
            I.e. do not put the first line immediately after the triple quotes; instead
            put the first line of text on a new line.

        Parameters
        ----------
        value : int
            Integer value of the cell type.

        _doc : str, optional
            Short description of this cell type. Typically a single line but no more
            than 3-4 lines. Should only include a general description and no technical
            details or cell connectivity.

        _example : str, optional
            Name of the example for this cell type in `pyvista.examples.cell.<NAME>`.
            When specified, the first figure from this example is used as the image for
            the cell.

        _variable_points: bool, optional
            Override the value shown for this cell type's `Points` badge. May be
            useful for composite cells (e.g. POLY_LINE or POLY_VERTEX) where a value
            of ``0`` may otherwise be shown. By default, the value from ``vtk_class``
            is used.

        _variable_edges: bool, optional
            Override the value shown for this cell type's `Edges` badge. May be
            useful for composite cells (e.g. POLY_LINE or POLY_VERTEX) where a value
            of ``0`` may otherwise be shown. By default, the value from ``vtk_class``
            is used.

        _variable_faces: bool, optional
            Override the value shown for this cell type's `Faces` badge. May be
            useful for composite cells (e.g. POLY_LINE or POLY_VERTEX) where a value
            of ``0`` may otherwise be shown. By default, the value from ``vtk_class``
            is used.

        """
        self = int.__new__(cls, value)
        self._value_ = value
        self._example = _example
        self.__doc__ = ''

        # Get the vtk class associated with this cell type. For simplicity, skip abstract types
        # even though some actually do have a corresponding class. Among other things, skipping
        # abstract types helps avoid the need to work around this VTK bug: https://gitlab.kitware.com/vtk/vtk/-/issues/19988
        _vtk_class_name = _vtk.vtkCellTypeUtilities.GetClassNameFromTypeId(value)
        if vtk_version_info < (9, 6, 0):
            # Fix bug with older VTK where higher order Triangle/Quadrilateral are swapped
            _vtk_class_name = {
                'vtkBezierTriangle': 'vtkBezierQuadrilateral',
                'vtkBezierQuadrilateral': 'vtkBezierTriangle',
                'vtkLagrangeTriangle': 'vtkLagrangeQuadrilateral',
                'vtkLagrangeQuadrilateral': 'vtkLagrangeTriangle',
            }.get(_vtk_class_name, _vtk_class_name)
        self._vtk_class = (
            None
            if _vtk_class_name.startswith(('vtkParametric', 'vtkHigherOrder'))  # Abstract
            or _vtk_class_name in ('vtkLagrangePyramid', 'vtkBezierPyramid')  # Missing vtk class
            or _vtk_class_name == 'UnknownClass'  # Abstract/missing classes in vtk 9.7.0
            else getattr(_vtk, _vtk_class_name)
        )

        # Set cell type properties using vtkCellTypeUtilities
        self._dimension = cast('_Dimension', _vtk.vtkCellTypeUtilities.GetDimension(value))
        self._is_linear = bool(_vtk.vtkCellTypeUtilities.IsLinear(value))
        if value == _vtk.VTK_HEXAGONAL_PRISM:
            # Need to fix https://gitlab.kitware.com/vtk/vtk/-/issues/19988#note_1786788
            self._is_linear = True

        # Set properties that require instantiating the class.
        # Assume primary cell with -1 values for points/edges/faces for abstract classes
        self._n_points = -1
        self._n_edges = -1
        self._n_faces = -1
        self._is_composite = False
        if self._vtk_class is not None:
            cell = self._vtk_class()
            self._is_composite = not bool(cell.IsPrimaryCell())

            # Use -1 to denote the cell has variable points/edges/faces
            self._n_points = -1 if _variable_points else cell.GetNumberOfPoints()
            self._n_edges = -1 if _variable_edges else cell.GetNumberOfEdges()
            self._n_faces = -1 if _variable_faces else cell.GetNumberOfFaces()

        _doc = textwrap.dedent(_doc).strip()

        # Generate cell type documentation if specified
        if self._vtk_class or _doc or _example:
            _badges = ''
            if self._vtk_class:
                _linear_badge = _generate_linear_badge(self._is_linear)
                _composite_badge = _generate_composite_badge(self._is_composite)
                _dimension_badge = _generate_dimension_badge(self._dimension)

                _points = 'variable' if _variable_points else self._n_points
                _points_badge = _generate_points_badge(_points)  # type: ignore[arg-type]

                _edges = 'variable' if _variable_edges else self._n_edges
                _edges_badge = _generate_edges_badge(_edges)  # type: ignore[arg-type]

                _faces = 'variable' if _variable_faces else self._n_faces
                _faces_badge = _generate_faces_badge(_faces)  # type: ignore[arg-type]

                _badges = _indent_paragraph(
                    f'{_linear_badge} {_composite_badge} {_dimension_badge}\n'
                    f'{_points_badge} {_edges_badge} {_faces_badge}',
                    level=2,
                )

                # Add additional references to VTK docs
                _vtk_class_ref = f':vtk:`{self._vtk_class.__name__}`'
                _see_also = f'See also {_vtk_class_ref}.'
                _doc += f'\n\n{_see_also}'

            _doc = _indent_paragraph(_doc, level=2)

            self.__doc__ += (
                _GRID_TEMPLATE_NO_IMAGE.format(_badges, _doc)
                if (_example is None or _example == 'Empty')
                else _GRID_TEMPLATE_WITH_IMAGE.format(
                    _example, _example, PLACEHOLDER, _badges, _doc
                )
            )

        return self

    @property
    def vtk_class(self) -> type[_vtk.vtkCell] | None:  # numpydoc ignore=RT01
        """Return the :vtk:`vtkCell` class associated with this cell type.

        Only concrete VTK classes that can instantiated are returned. ``None`` is returned
        if the cell type has no corresponding VTK class or if the class is abstract.

        .. versionadded:: 0.48

        See Also
        --------
        pyvista.Cell

        Examples
        --------
        Get the VTK class associated with the :attr:`TRIANGLE` cell type.

        >>> import pyvista as pv
        >>> pv.CellType.TRIANGLE.vtk_class
        <class 'vtkmodules.vtkCommonDataModel.vtkTriangle'>

        Abstract cell types like ``HIGHER_ORDER_HEXAHEDRON`` return ``None``.

        >>> pv.CellType.HIGHER_ORDER_HEXAHEDRON.vtk_class is None
        True

        """
        return self._vtk_class

    @property
    def dimension(self) -> _Dimension:  # numpydoc ignore=RT01
        """Return this cell type's dimension.

        .. versionadded:: 0.48

        See Also
        --------
        pyvista.Cell.dimension
        dimension_map

        Examples
        --------
        :attr:`VERTEX` is 0D.

        >>> import pyvista as pv
        >>> pv.CellType.VERTEX.dimension
        0

        :attr:`LINE` is 1D.

        >>> pv.CellType.LINE.dimension
        1

        :attr:`TRIANGLE` is 2D.

        >>> pv.CellType.TRIANGLE.dimension
        2

        :attr:`HEXAHEDRON` is 3D.

        >>> pv.CellType.HEXAHEDRON.dimension
        3

        """
        return self._dimension

    @property
    def is_linear(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if this cell type is linear.

        .. versionadded:: 0.48

        See Also
        --------
        is_composite
        pyvista.Cell.is_linear

        Examples
        --------
        :attr:`HEXAHEDRON` is linear.

        >>> import pyvista as pv
        >>> pv.CellType.HEXAHEDRON.is_linear
        True

        :attr:`QUADRATIC_HEXAHEDRON` is not linear.

        >>> pv.CellType.QUADRATIC_HEXAHEDRON.is_linear
        False

        """
        return self._is_linear

    @property
    def is_composite(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if this cell type is composed of multiple smaller cells.

        .. versionadded:: 0.48

        See Also
        --------
        is_linear

        Examples
        --------
        :attr:`VERTEX` is a primary cell and is not composite.

        >>> import pyvista as pv
        >>> pv.CellType.VERTEX.is_composite
        False

        :attr:`POLY_VERTEX` `is` composite since it's composed of multiple vertex cells.

        >>> pv.CellType.POLY_VERTEX.is_composite
        True

        """
        return self._is_composite

    @property
    def n_points(self) -> int:  # numpydoc ignore=RT01
        """Return the number of points defined by this cell type.

        .. versionadded:: 0.48

        See Also
        --------
        n_edges, n_faces
        pyvista.Cell.n_points

        Raises
        ------
        ValueError
            If the number of points cannot be determined without a concrete instance.
            Composite cells, higher order cells, polygon, and polyhedron will all raise an error.

        Examples
        --------
        :attr:`LINE` has a fixed number of points.

        >>> import pyvista as pv
        >>> pv.CellType.LINE.n_points
        2

        :attr:`POLY_LINE` has a variable number of points and raises a ValueError.

        >>> try:
        ...     pv.CellType.POLY_LINE.n_points
        ... except ValueError as error:
        ...     print(error)
        Cannot determine number of points for 'POLY_LINE' without a concrete cell instance.

        """
        if (n_points := self._n_points) == -1:
            msg = (
                f'Cannot determine number of points for {self.name!r} '
                f'without a concrete cell instance.'
            )
            raise ValueError(msg)
        return n_points

    @property
    def n_edges(self) -> int:  # numpydoc ignore=RT01
        """Return the number of edges defined by this cell type.

        .. versionadded:: 0.48

        See Also
        --------
        n_points, n_faces
        pyvista.Cell.n_edges

        Raises
        ------
        ValueError
            If the number of edges cannot be determined without a concrete instance.
            Polygon, polyhedron, triangle strip, and convex point set will all raise an error.

        Examples
        --------
        :attr:`QUAD` has a fixed number of edges.

        >>> import pyvista as pv
        >>> pv.CellType.QUAD.n_edges
        4

        :attr:`POLYGON` has a variable number of edges and raises a ValueError.

        >>> try:
        ...     pv.CellType.POLYGON.n_edges
        ... except ValueError as error:
        ...     print(error)
        Cannot determine number of edges for 'POLYGON' without a concrete cell instance.

        Zero- and one-dimensional cell types have no edges.

        >>> pv.CellType.VERTEX.n_edges
        0

        >>> pv.CellType.LINE.n_edges
        0

        """
        if (n_edges := self._n_edges) == -1:
            msg = (
                f'Cannot determine number of edges for {self.name!r} '
                f'without a concrete cell instance.'
            )
            raise ValueError(msg)
        return n_edges

    @property
    def n_faces(self) -> int:  # numpydoc ignore=RT01
        """Return the number of faces defined by this cell type.

        .. versionadded:: 0.48

        See Also
        --------
        n_points, n_edges
        pyvista.Cell.n_faces

        Raises
        ------
        ValueError
            If the number of faces cannot be determined without a concrete instance.
            Polyhedron and convex point set will raise an error.

        Examples
        --------
        :attr:`WEDGE` has a fixed number of faces.

        >>> import pyvista as pv
        >>> pv.CellType.WEDGE.n_faces
        5

        :attr:`POLYHEDRON` has a variable number of faces and raises a ValueError.

        >>> try:
        ...     pv.CellType.POLYHEDRON.n_faces
        ... except ValueError as error:
        ...     print(error)
        Cannot determine number of faces for 'POLYHEDRON' without a concrete cell instance.

        Only three-dimensional cell types have faces.

        >>> pv.CellType.LINE.n_faces
        0

        >>> pv.CellType.QUAD.n_faces
        0

        """
        if (n_faces := self._n_faces) == -1:
            msg = (
                f'Cannot determine number of faces for {self.name!r} '
                f'without a concrete cell instance.'
            )
            raise ValueError(msg)
        return n_faces

    @_classproperty
    def dimension_map(cls) -> dict[Literal[0, 1, 2, 3], frozenset[CellType]]:  # noqa: N805
        """Return a mapping with sets for all cell types grouped by topological dimension.

        The groupings are derived from each member's :attr:`dimension`.

        .. versionadded:: 0.48

        Returns
        -------
        dict
            Dictionary with cell dimensions ``0``, ``1``, ``2,``, ``3`` as keys, and the respective
            :class:`CellType` members as values.

        See Also
        --------
        dimension
        pyvista.DataSet.max_cell_dimensionality
        pyvista.DataSet.min_cell_dimensionality
        pyvista.DataSet.distinct_cell_types

        Examples
        --------
        Common 2D cell types such as :attr:`TRIANGLE` and :attr:`TRIANGLE_STRIP`
        are in the 2D grouping.

        >>> import pyvista as pv
        >>> two_d = pv.CellType.dimension_map[2]
        >>> pv.CellType.TRIANGLE in two_d
        True
        >>> pv.CellType.TRIANGLE_STRIP in two_d
        True

        Check whether a mesh contains only 2D cells by combining this with
        :attr:`~pyvista.DataSet.distinct_cell_types`.

        >>> mesh = pv.Sphere()
        >>> mesh.distinct_cell_types <= pv.CellType.dimension_map[2]
        True

        Show all 0D cell types.

        >>> sorted(pv.CellType.dimension_map[0])  # doctest: +NORMALIZE_WHITESPACE
        [<CellType.EMPTY_CELL: 0>,
         <CellType.VERTEX: 1>,
         <CellType.POLY_VERTEX: 2>]

        Show all 1D cell types.

        >>> sorted(pv.CellType.dimension_map[1])  # doctest: +NORMALIZE_WHITESPACE
        [<CellType.LINE: 3>,
         <CellType.POLY_LINE: 4>,
         <CellType.QUADRATIC_EDGE: 21>,
         <CellType.CUBIC_LINE: 35>,
         <CellType.LAGRANGE_CURVE: 68>,
         <CellType.BEZIER_CURVE: 75>]

        Show all 2D cell types.

        >>> sorted(pv.CellType.dimension_map[2])  # doctest: +NORMALIZE_WHITESPACE
        [<CellType.TRIANGLE: 5>,
         <CellType.TRIANGLE_STRIP: 6>,
         <CellType.POLYGON: 7>,
         <CellType.PIXEL: 8>,
         <CellType.QUAD: 9>,
         <CellType.QUADRATIC_TRIANGLE: 22>,
         <CellType.QUADRATIC_QUAD: 23>,
         <CellType.BIQUADRATIC_QUAD: 28>,
         <CellType.QUADRATIC_LINEAR_QUAD: 30>,
         <CellType.BIQUADRATIC_TRIANGLE: 34>,
         <CellType.QUADRATIC_POLYGON: 36>,
         <CellType.HIGHER_ORDER_TRIANGLE: 61>,
         <CellType.HIGHER_ORDER_QUAD: 62>,
         <CellType.LAGRANGE_TRIANGLE: 69>,
         <CellType.LAGRANGE_QUADRILATERAL: 70>,
         <CellType.BEZIER_TRIANGLE: 76>,
         <CellType.BEZIER_QUADRILATERAL: 77>]

        Show all 3D cell types.

        >>> sorted(pv.CellType.dimension_map[3])  # doctest: +NORMALIZE_WHITESPACE
        [<CellType.TETRA: 10>,
         <CellType.VOXEL: 11>,
         <CellType.HEXAHEDRON: 12>,
         <CellType.WEDGE: 13>,
         <CellType.PYRAMID: 14>,
         <CellType.PENTAGONAL_PRISM: 15>,
         <CellType.HEXAGONAL_PRISM: 16>,
         <CellType.QUADRATIC_TETRA: 24>,
         <CellType.QUADRATIC_HEXAHEDRON: 25>,
         <CellType.QUADRATIC_WEDGE: 26>,
         <CellType.QUADRATIC_PYRAMID: 27>,
         <CellType.TRIQUADRATIC_HEXAHEDRON: 29>,
         <CellType.QUADRATIC_LINEAR_WEDGE: 31>,
         <CellType.BIQUADRATIC_QUADRATIC_WEDGE: 32>,
         <CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON: 33>,
         <CellType.TRIQUADRATIC_PYRAMID: 37>,
         <CellType.CONVEX_POINT_SET: 41>,
         <CellType.POLYHEDRON: 42>,
         <CellType.PARAMETRIC_CURVE: 51>,
         <CellType.PARAMETRIC_SURFACE: 52>,
         <CellType.PARAMETRIC_TRI_SURFACE: 53>,
         <CellType.PARAMETRIC_QUAD_SURFACE: 54>,
         <CellType.PARAMETRIC_TETRA_REGION: 55>,
         <CellType.PARAMETRIC_HEX_REGION: 56>,
         <CellType.HIGHER_ORDER_EDGE: 60>,
         <CellType.HIGHER_ORDER_POLYGON: 63>,
         <CellType.HIGHER_ORDER_TETRAHEDRON: 64>,
         <CellType.HIGHER_ORDER_WEDGE: 65>,
         <CellType.HIGHER_ORDER_PYRAMID: 66>,
         <CellType.HIGHER_ORDER_HEXAHEDRON: 67>,
         <CellType.LAGRANGE_TETRAHEDRON: 71>,
         <CellType.LAGRANGE_HEXAHEDRON: 72>,
         <CellType.LAGRANGE_WEDGE: 73>,
         <CellType.LAGRANGE_PYRAMID: 74>,
         <CellType.BEZIER_TETRAHEDRON: 78>,
         <CellType.BEZIER_HEXAHEDRON: 79>,
         <CellType.BEZIER_WEDGE: 80>,
         <CellType.BEZIER_PYRAMID: 81>]

        """
        if not _DIMENSION_MAP:
            # Populate dict on first access
            for dimension in get_args(_Dimension):
                _DIMENSION_MAP[dimension] = frozenset(
                    celltype for celltype in CellType if celltype.dimension == dimension
                )

        # Return copy to avoid users mutating it
        return dict(_DIMENSION_MAP)

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
