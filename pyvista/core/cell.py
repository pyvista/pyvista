"""Define types of cells."""
from __future__ import annotations

from enum import IntEnum
from typing import Generator, List, Tuple

import numpy as np

import pyvista
from pyvista import _vtk


class Cell(_vtk.VTKObjectWrapper):
    """Wrapping of vtkCell.

    This class provide the capability to access a given cell topology.

    Parameters
    ----------
    vtkobject : vtk.vtkCell
        The vtk object to wrap as Cell, that must be of vtk.vtkCell type.

    Examples
    --------
    Get the 0-th cell information

    >>> import pyvista
    >>> mesh = pyvista.Sphere()
    >>> mesh.cell[0] # doctest: +SKIP
    Cell (0x7f2881968f40)
      Type:	CellType.TRIANGLE
      Linear:       True
      Dimension:	2
      N Points:	    3
      N Faces:	    0
      N Edges:	    3
      X Bounds:	    -5.406e-02, -5.551e-17
      Y Bounds:	    0.000e+00, 1.124e-02
      Z Bounds:	    -5.000e-01, -4.971e-01
    """

    def __init__(self, vtkobject: _vtk.vtkCell) -> None:
        """Init the cell object with a _vtk.vtkCell."""
        if not isinstance(vtkobject, _vtk.vtkCell):
            msg = f"`vtkobject` must be of vtkCell type (got {type(vtkobject)}) instead"
            raise TypeError(msg)
        super().__init__(vtkobject=vtkobject)

    @property
    def type(self) -> CellType:
        """Get the cell type from the enum `CellType`.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].type
        <CellType.TRIANGLE: 5>
        """
        return CellType(self.GetCellType())

    @property
    def is_linear(self) -> bool:
        """Get if the cell is linear.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].is_linear
        True
        """
        return bool(self.IsLinear())

    @property
    def dimension(self) -> int:
        """Get the cell dimension. For example, 2 for a triangle and 3 for a tetrahedron.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].dimension
        2
        """
        return self.GetCellDimension()

    @property
    def n_points(self) -> int:
        """Get the number of points building the cell.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].n_points
        3
        """
        return self.GetNumberOfPoints()

    @property
    def n_faces(self) -> int:
        """Get the number of faces building the cell.

        Examples
        --------
        >>> from pyvista.examples.cells import Tetrahedron
        >>> mesh = Tetrahedron()
        >>> mesh.cell[0].n_faces
        4
        """
        return self.GetNumberOfFaces()

    @property
    def n_edges(self) -> int:
        """Get the number of edges building the cell.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].n_edges
        3
        """
        return self.GetNumberOfEdges()

    @property
    def point_ids(self) -> List[int]:
        """Get the point ids building the cell.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].point_ids
        [2, 30, 0]
        """
        point_ids = self.GetPointIds()
        return [point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds())]

    @property
    def points(self) -> np.ndarray:
        """Get the point coordinates of the cell.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].points
        array([[-5.40595092e-02,  0.00000000e+00, -4.97068971e-01],
               [-5.28781787e-02,  1.12396041e-02, -4.97068971e-01],
               [-5.55111512e-17,  0.00000000e+00, -5.00000000e-01]])
        """
        # A copy of the points must be returned to avoid overlapping them since the
        # `vtk.vtkExplicitStructuredGrid.GetCell` is an override method.
        points = _vtk.vtk_to_numpy(self.GetPoints().GetData())
        return points.copy()

    @property
    def edges(self) -> Generator[Cell, None, None]:
        """Get an iterator of edges building the cell.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].edges[0] # doctest:+SKIP
        Cell (0x7f4c1b044b20)
          Type:	    CellType.LINE
          Linear:	True
          Dimension:	1
          N Points:	    2
          N Faces:	    0
          N Edges:	    0
          X Bounds:	    -1.075e-01, -1.051e-01
          Y Bounds:	    -2.235e-02, 0.000e+00
          Z Bounds:	    4.883e-01, 4.883e-01
        """
        for i in range(self.n_edges):
            yield self.get_edge(i)

    def get_edge(self, i) -> Cell:
        """Get the i-th edge building the cell."""
        return Cell(self.GetEdge(i))

    @property
    def faces(self) -> Generator[Cell, None, None]:
        """Get an iterator of cell faces.

        Examples
        --------
        >>> from pyvista.examples.cells import Tetrahedron
        >>> mesh = Tetrahedron()
        >>> mesh.cell[0].faces[0] # doctest:+SKIP
        Cell (0x7f4c1b044190)
          Type:	CellType.TRIANGLE
          Linear:	True
          Dimension:	2
          N Points:	3
          N Faces:	0
          N Edges:	3
          X Bounds:	-1.000e+00, 1.000e+00
          Y Bounds:	-1.000e+00, 1.000e+00
          Z Bounds:	-1.000e+00, 1.000e+00
        """
        for i in range(self.n_faces):
            yield self.get_face(i)

    def get_face(self, i) -> Cell:
        """Get the i-th face building the cell."""
        return Cell(self.GetFace(i))

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get the cell bounds [xmin, xmax, ymin, ymax, zmin, zmax].

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.cell[0].bounds
        (-0.05405950918793678, -5.551115123125783e-17, 0.0, 0.011239604093134403, -0.5, -0.49706897139549255)
        """
        return self.GetBounds()

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(("Type", str(self.type), "{}" * len(str(self.type))))
        attrs.append(("Linear", self.is_linear, "{}"))
        attrs.append(("Dimension", self.dimension, "{}"))
        attrs.append(("N Points", self.n_points, "{}"))
        attrs.append(("N Faces", self.n_faces, "{}"))
        attrs.append(("N Edges", self.n_edges, "{}"))
        bds = self.bounds
        fmt = f"{pyvista.FLOAT_FORMAT}, {pyvista.FLOAT_FORMAT}"
        attrs.append(("X Bounds", (bds[0], bds[1]), fmt))
        attrs.append(("Y Bounds", (bds[2], bds[3]), fmt))
        attrs.append(("Z Bounds", (bds[4], bds[5]), fmt))

        return attrs

    def _str(self, html=False):
        """Return a string with the cell info."""
        fmt = ""
        if html:
            return "Not implemented yet"

        # Otherwise return a string that is Python console friendly
        fmt = f"{type(self).__name__} ({hex(id(self))})\n"
        # now make a call on the object to get its attributes as a list of len 2 tuples
        row = "  {}:\t{}\n"
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0], attr[2].format(attr[1]))
        return fmt

    def __repr__(self):
        """Return the object representation."""
        return self._str(html=False)


class CellType(IntEnum):
    """Define types of cells.

    Warnings
    --------
    The following types need ``vtk >=v9.0.0`` (See `Implementation of vtkTriQuadraticPyramid cell
    <https://gitlab.kitware.com/vtk/vtk/-/merge_requests/8295>`_ and `Add Bezier cell types <https://gitlab.kitware.com/vtk/vtk/-/merge_requests/6055>`_ ).
    * TRIQUADRATIC_PYRAMID
    * BEZIER_TRIANGLE
    * BEZIER_QUADRILATERAL
    * BEZIER_TETRAHEDRON
    * BEZIER_HEXAHEDRON
    * BEZIER_WEDGE
    * BEZIER_PYRAMID

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
    if hasattr(_vtk, "VTK_BEZIER_CURVE"):
        BEZIER_CURVE = _vtk.VTK_BEZIER_CURVE
    if hasattr(_vtk, "VTK_BEZIER_TRIANGLE"):
        BEZIER_TRIANGLE = _vtk.VTK_BEZIER_TRIANGLE
    if hasattr(_vtk, "VTK_BEZIER_QUADRILATERAL"):
        BEZIER_QUADRILATERAL = _vtk.VTK_BEZIER_QUADRILATERAL
    if hasattr(_vtk, "VTK_BEZIER_TETRAHEDRON"):
        BEZIER_TETRAHEDRON = _vtk.VTK_BEZIER_TETRAHEDRON
    if hasattr(_vtk, "VTK_BEZIER_HEXAHEDRON"):
        BEZIER_HEXAHEDRON = _vtk.VTK_BEZIER_HEXAHEDRON
    if hasattr(_vtk, "VTK_BEZIER_WEDGE"):
        BEZIER_WEDGE = _vtk.VTK_BEZIER_WEDGE
    if hasattr(_vtk, "VTK_BEZIER_PYRAMID"):
        BEZIER_PYRAMID = _vtk.VTK_BEZIER_PYRAMID
