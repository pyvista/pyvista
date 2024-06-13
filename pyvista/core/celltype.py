"""Define types of cells."""

from __future__ import annotations

from enum import IntEnum

from . import _vtk_core as _vtk


class CellType(IntEnum):
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

    # Linear cells
    EMPTY_CELL = _vtk.VTK_EMPTY_CELL
    VERTEX = _vtk.VTK_VERTEX
    POLY_VERTEX = _vtk.VTK_POLY_VERTEX
    LINE = _vtk.VTK_LINE
    POLY_LINE = _vtk.VTK_POLY_LINE
    TRIANGLE = _vtk.VTK_TRIANGLE
    TRIANGLE_STRIP = _vtk.VTK_TRIANGLE_STRIP
    POLYGON = _vtk.VTK_POLYGON
    PIXEL = _vtk.VTK_PIXEL
    QUAD = _vtk.VTK_QUAD
    TETRA = _vtk.VTK_TETRA
    VOXEL = _vtk.VTK_VOXEL
    HEXAHEDRON = _vtk.VTK_HEXAHEDRON
    WEDGE = _vtk.VTK_WEDGE
    PYRAMID = _vtk.VTK_PYRAMID
    PENTAGONAL_PRISM = _vtk.VTK_PENTAGONAL_PRISM
    HEXAGONAL_PRISM = _vtk.VTK_HEXAGONAL_PRISM

    # Quadratic, isoparametric cells
    QUADRATIC_EDGE = _vtk.VTK_QUADRATIC_EDGE
    QUADRATIC_TRIANGLE = _vtk.VTK_QUADRATIC_TRIANGLE
    QUADRATIC_QUAD = _vtk.VTK_QUADRATIC_QUAD
    QUADRATIC_POLYGON = _vtk.VTK_QUADRATIC_POLYGON
    QUADRATIC_TETRA = _vtk.VTK_QUADRATIC_TETRA
    QUADRATIC_HEXAHEDRON = _vtk.VTK_QUADRATIC_HEXAHEDRON
    QUADRATIC_WEDGE = _vtk.VTK_QUADRATIC_WEDGE
    QUADRATIC_PYRAMID = _vtk.VTK_QUADRATIC_PYRAMID
    BIQUADRATIC_QUAD = _vtk.VTK_BIQUADRATIC_QUAD
    TRIQUADRATIC_HEXAHEDRON = _vtk.VTK_TRIQUADRATIC_HEXAHEDRON
    if hasattr(_vtk, "VTK_TRIQUADRATIC_PYRAMID"):
        TRIQUADRATIC_PYRAMID = _vtk.VTK_TRIQUADRATIC_PYRAMID
    QUADRATIC_LINEAR_QUAD = _vtk.VTK_QUADRATIC_LINEAR_QUAD
    QUADRATIC_LINEAR_WEDGE = _vtk.VTK_QUADRATIC_LINEAR_WEDGE
    BIQUADRATIC_QUADRATIC_WEDGE = _vtk.VTK_BIQUADRATIC_QUADRATIC_WEDGE
    BIQUADRATIC_QUADRATIC_HEXAHEDRON = _vtk.VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON
    BIQUADRATIC_TRIANGLE = _vtk.VTK_BIQUADRATIC_TRIANGLE

    # Cubic, isoparametric cell
    CUBIC_LINE = _vtk.VTK_CUBIC_LINE

    # Special class of cells formed by convex group of points
    CONVEX_POINT_SET = _vtk.VTK_CONVEX_POINT_SET

    # Polyhedron cell (consisting of polygonal faces)
    POLYHEDRON = _vtk.VTK_POLYHEDRON

    # Higher order cells in parametric form
    PARAMETRIC_CURVE = _vtk.VTK_PARAMETRIC_CURVE
    PARAMETRIC_SURFACE = _vtk.VTK_PARAMETRIC_SURFACE
    PARAMETRIC_TRI_SURFACE = _vtk.VTK_PARAMETRIC_TRI_SURFACE
    PARAMETRIC_QUAD_SURFACE = _vtk.VTK_PARAMETRIC_QUAD_SURFACE
    PARAMETRIC_TETRA_REGION = _vtk.VTK_PARAMETRIC_TETRA_REGION
    PARAMETRIC_HEX_REGION = _vtk.VTK_PARAMETRIC_HEX_REGION

    # Higher order cells
    HIGHER_ORDER_EDGE = _vtk.VTK_HIGHER_ORDER_EDGE
    HIGHER_ORDER_TRIANGLE = _vtk.VTK_HIGHER_ORDER_TRIANGLE
    HIGHER_ORDER_QUAD = _vtk.VTK_HIGHER_ORDER_QUAD
    HIGHER_ORDER_POLYGON = _vtk.VTK_HIGHER_ORDER_POLYGON
    HIGHER_ORDER_TETRAHEDRON = _vtk.VTK_HIGHER_ORDER_TETRAHEDRON
    HIGHER_ORDER_WEDGE = _vtk.VTK_HIGHER_ORDER_WEDGE
    HIGHER_ORDER_PYRAMID = _vtk.VTK_HIGHER_ORDER_PYRAMID
    HIGHER_ORDER_HEXAHEDRON = _vtk.VTK_HIGHER_ORDER_HEXAHEDRON

    # Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
    LAGRANGE_CURVE = _vtk.VTK_LAGRANGE_CURVE
    LAGRANGE_TRIANGLE = _vtk.VTK_LAGRANGE_TRIANGLE
    LAGRANGE_QUADRILATERAL = _vtk.VTK_LAGRANGE_QUADRILATERAL
    LAGRANGE_TETRAHEDRON = _vtk.VTK_LAGRANGE_TETRAHEDRON
    LAGRANGE_HEXAHEDRON = _vtk.VTK_LAGRANGE_HEXAHEDRON
    LAGRANGE_WEDGE = _vtk.VTK_LAGRANGE_WEDGE
    LAGRANGE_PYRAMID = _vtk.VTK_LAGRANGE_PYRAMID

    # Arbitrary order Bezier elements (formulated separated from generic higher order cells)
    BEZIER_CURVE = _vtk.VTK_BEZIER_CURVE
    BEZIER_TRIANGLE = _vtk.VTK_BEZIER_TRIANGLE
    BEZIER_QUADRILATERAL = _vtk.VTK_BEZIER_QUADRILATERAL
    BEZIER_TETRAHEDRON = _vtk.VTK_BEZIER_TETRAHEDRON
    BEZIER_HEXAHEDRON = _vtk.VTK_BEZIER_HEXAHEDRON
    BEZIER_WEDGE = _vtk.VTK_BEZIER_WEDGE
    BEZIER_PYRAMID = _vtk.VTK_BEZIER_PYRAMID
