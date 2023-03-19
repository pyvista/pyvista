"""Contains the pyvista.Cell class."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

import pyvista
from pyvista import _vtk

from .celltype import CellType
from .dataset import DataObject


class Cell(_vtk.vtkGenericCell, DataObject):
    """Wrapping of vtkCell.

    This class provides the capability to access a given cell topology and can
    be useful when walking through a cell's individual faces or investigating
    cell properties.

    Parameters
    ----------
    vtk_cell : vtk.vtkCell, optional
        The vtk object to wrap as Cell, that must be of ``vtk.vtkCell`` type.

    cell_type : int, optional
        VTK cell type. Determined from ``vtk_cell`` if not input.

    deep : bool, default: False
        Perform a deep copy of the original cell.

    Notes
    -----
    Accessing individual cells from a :class:`pyvista.DataSet` using this class
    will be much slower than accessing bulk data from the
    :attr:`pyvista.PolyData.faces` or :attr:`pyvista.UnstructuredGrid.cells` attributes.

    Also note that the cell object is a deep copy of the original cell and
    is unassociated with the original cell. Changing any data of
    that cell (for example, :attr:`pyvista.Cell.points`) will not change the original dataset.

    Examples
    --------
    Get the 0-th cell from a :class:`pyvista.PolyData`.

    >>> import pyvista
    >>> mesh = pyvista.Sphere()
    >>> cell = mesh.get_cell(0)
    >>> cell  # doctest: +SKIP
    Cell (0x7fa760075a10)
      Type:       <CellType.TRIANGLE: 5>
      Linear:     True
      Dimension:  2
      N Points:	  3
      N Faces:    0
      N Edges:    3
      X Bounds:   -5.406e-02, -5.551e-17
      Y Bounds:	  0.000e+00, 1.124e-02
      Z Bounds:   -5.000e-01, -4.971e-01

    Get the 0-th cell from a :class:`pyvista.UnstructuredGrid`.

    >>> from pyvista import examples
    >>> mesh = examples.load_hexbeam()
    >>> cell = mesh.get_cell(0)
    >>> cell  # doctest: +SKIP
    Cell (0x7fdc71a3c210)
      Type:       <CellType.HEXAHEDRON: 12>
      Linear:     True
      Dimension:  3
      N Points:   8
      N Faces:    6
      N Edges:    12
      X Bounds:   0.000e+00, 5.000e-01
      Y Bounds:   0.000e+00, 5.000e-01
      Z Bounds:   0.000e+00, 5.000e-01

    """

    def __init__(self, vtk_cell=None, cell_type=None, deep=False):
        """Initialize the cell."""
        super().__init__()
        if vtk_cell is not None:
            if not isinstance(vtk_cell, _vtk.vtkCell):
                raise TypeError(f'`vtk_cell` must be a vtkCell, not {type(vtk_cell)}')
            # cell type must be set first before deep or shallow copy
            if cell_type is None:
                self.SetCellType(vtk_cell.GetCellType())
            else:
                self.SetCellType(cell_type)

            if deep:
                self.DeepCopy(vtk_cell)
            else:
                self.ShallowCopy(vtk_cell)

    @property
    def type(self) -> CellType:
        """Get the cell type from the enum :class:`pyvista.CellType`.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.get_cell(0).type
        <CellType.TRIANGLE: 5>
        """
        return CellType(self.GetCellType())

    @property
    def is_linear(self) -> bool:
        """Return if the cell is linear.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.get_cell(0).is_linear
        True

        """
        return bool(self.IsLinear())

    def plot(self, **kwargs):
        """Plot this cell.

        Parameters
        ----------
        **kwargs : dict, optional
            See :func:`pyvista.plot` for a description of the optional keyword
            arguments.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_hexbeam()
        >>> cell = mesh.get_cell(0)
        >>> cell.plot()

        """
        self.cast_to_unstructured_grid().plot(**kwargs)

    def cast_to_unstructured_grid(self) -> pyvista.UnstructuredGrid:
        """Cast this cell to an unstructured grid.

        Returns
        -------
        pyvista.UnstructuredGrid
            This cell cast to a :class:`pyvista.UnstructuredGrid`.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_hexbeam()
        >>> cell = mesh.get_cell(0)
        >>> grid = cell.cast_to_unstructured_grid()
        >>> grid  # doctest: +SKIP
        UnstructuredGrid (0x7f9383619540)
          N Cells:      1
          N Points:     8
          X Bounds:     0.000e+00, 5.000e-01
          Y Bounds:     0.000e+00, 5.000e-01
          Z Bounds:     0.000e+00, 5.000e-01
          N Arrays:     0

        """
        return pyvista.UnstructuredGrid(
            [len(self.point_ids)] + list(range(len(self.point_ids))),
            [int(self.type)],
            self.points.copy(),
        )

    @property
    def dimension(self) -> int:
        """Return the cell dimension.

        This returns the dimensionality of the cell. For example, 1 for an edge,
        2 for a triangle, and 3 for a tetrahedron.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.get_cell(0).dimension
        2
        """
        return self.GetCellDimension()

    @property
    def n_points(self) -> int:
        """Get the number of points composing the cell.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.get_cell(0).n_points
        3
        """
        return self.GetNumberOfPoints()

    @property
    def n_faces(self) -> int:
        """Get the number of faces composing the cell.

        Examples
        --------
        >>> from pyvista.examples.cells import Tetrahedron
        >>> mesh = Tetrahedron()
        >>> mesh.get_cell(0).n_faces
        4
        """
        return self.GetNumberOfFaces()

    @property
    def n_edges(self) -> int:
        """Get the number of edges composing the cell.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.get_cell(0).n_edges
        3
        """
        return self.GetNumberOfEdges()

    @property
    def point_ids(self) -> List[int]:
        """Get the point IDs composing the cell.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.get_cell(0).point_ids
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
        >>> mesh.get_cell(0).points
        array([[-5.40595092e-02,  0.00000000e+00, -4.97068971e-01],
               [-5.28781787e-02,  1.12396041e-02, -4.97068971e-01],
               [-5.55111512e-17,  0.00000000e+00, -5.00000000e-01]])
        """
        return _vtk.vtk_to_numpy(self.GetPoints().GetData())

    def get_edge(self, index: int) -> Cell:
        """Get the i-th edge composing the cell.

        Parameters
        ----------
        index : int
            Edge ID.

        Returns
        -------
        pyvista.Cell
            Edge given by ``index``.

        Examples
        --------
        Extract a single edge from a face and output the IDs of the edge
        points.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> cell = mesh.get_cell(0)
        >>> edge = cell.get_edge(0)
        >>> edge.point_ids
        [2, 30]

        """
        if index + 1 > self.n_edges:
            raise IndexError(f'Invalid index {index} for a cell with {self.n_edges} edges.')

        # must deep copy here as multiple sequental calls to GetEdge overwrite
        # the underlying pointer
        return Cell(self.GetEdge(index), deep=True)

    @property
    def edges(self) -> List[Cell]:
        """Return a list of edges composing the cell.

        >>> from pyvista.examples.cells import Hexahedron
        >>> mesh = Hexahedron()
        >>> cell = mesh.get_cell(0)
        >>> edges = cell.edges
        >>> len(edges)
        12

        """
        return [self.get_edge(i) for i in range(self.n_edges)]

    @property
    def faces(self) -> List[Cell]:
        """Return a list of faces composing the cell.

        >>> from pyvista.examples.cells import Tetrahedron
        >>> mesh = Tetrahedron()
        >>> cell = mesh.get_cell(0)
        >>> faces = cell.faces
        >>> len(faces)
        4

        """
        return [self.get_face(i) for i in range(self.n_faces)]

    def get_face(self, index: int) -> Cell:
        """Get the i-th face composing the cell.

        Parameters
        ----------
        index : int
            Face ID.

        Returns
        -------
        pyvista.Cell
            Face given by ``index``.

        Examples
        --------
        Return the face IDs composing the first face of an example tetrahedron.

        >>> from pyvista.examples.cells import Tetrahedron
        >>> mesh = Tetrahedron()
        >>> cell = mesh.get_cell(0)
        >>> face = cell.get_face(0)
        >>> face.point_ids
        [0, 1, 3]

        """
        # must deep copy here as sequental calls overwrite the underlying pointer
        if index + 1 > self.n_faces:
            raise IndexError(f'Invalid index {index} for a cell with {self.n_faces} faces.')

        # must deep copy here as multiple sequental calls to GetFace overwrite
        # the underlying pointer
        cell = self.GetFace(index)
        return Cell(cell, deep=True, cell_type=cell.GetCellType())

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get the cell bounds in ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.get_cell(0).bounds
        (-0.05405950918793678, -5.551115123125783e-17, 0.0, 0.011239604093134403, -0.5, -0.49706897139549255)
        """
        return self.GetBounds()

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(("Type", repr(self.type), "{}" * len(repr(self.type))))
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

    def __repr__(self) -> str:
        """Return the object representation."""
        return self.head(display=False, html=False)

    def __str__(self) -> str:
        """Return the object string representation."""
        return self.head(display=False, html=False)

    def copy(self, deep=True) -> Cell:
        """Return a copy of the cell.

        Parameters
        ----------
        deep : bool, optional
            When ``True`` makes a full copy of the cell.  When ``False``,
            performs a shallow copy where the new cell still references the
            original cell.

        Returns
        -------
        pyvista.Cell
            Deep or shallow copy of the cell.

        Examples
        --------
        Create a deep copy of the cell and demonstrate it is deep.

        >>> from pyvista.examples.cells import Tetrahedron
        >>> mesh = Tetrahedron()
        >>> cell = mesh.get_cell(0)
        >>> deep_cell = cell.copy(deep=True)
        >>> deep_cell.points[:] = 0
        >>> cell != deep_cell
        True

        Create a shallow copy of the cell and demonstrate it is shallow.

        >>> shallow_cell = cell.copy(deep=False)
        >>> shallow_cell.points[:] = 0
        >>> cell == shallow_cell
        True

        """
        return type(self)(self, deep=deep)
