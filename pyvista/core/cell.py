"""Contains the pyvista.Cell class."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union, cast

import numpy as np

import pyvista

from . import _vtk_core as _vtk
from ._typing_core import IntMatrix, IntVector, NumpyIntArray
from .celltype import CellType
from .dataset import DataObject
from .utilities.cells import ncells_from_cells, numpy_to_idarr


def _get_vtk_id_type():
    """Return the numpy datatype responding to ``vtk.vtkIdTypeArray``."""
    VTK_ID_TYPE_SIZE = _vtk.vtkIdTypeArray().GetDataTypeSize()
    if VTK_ID_TYPE_SIZE == 4:
        return np.int32
    elif VTK_ID_TYPE_SIZE == 8:
        return np.int64
    return np.int32


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

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
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
    def type(self) -> CellType:  # numpydoc ignore=RT01
        """Get the cell type from the enum :class:`pyvista.CellType`.

        Returns
        -------
        pyvista.CellType
            Type of cell.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.get_cell(0).type
        <CellType.TRIANGLE: 5>
        """
        return CellType(self.GetCellType())

    @property
    def is_linear(self) -> bool:  # numpydoc ignore=RT01
        """Return if the cell is linear.

        Returns
        -------
        bool
            If the cell is linear.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
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

    def cast_to_polydata(self) -> pyvista.PolyData:
        """Cast this cell to PolyData.

        Can only be used for 0D, 1D, or 2D cells.

        Returns
        -------
        pyvista.PolyData
            This cell cast to a :class:`pyvista.PolyData`.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_sphere()
        >>> cell = mesh.get_cell(0)
        >>> grid = cell.cast_to_polydata()
        >>> grid  # doctest: +SKIP
        PolyData (0x7f09ae437b80)
          N Cells:    1
          N Points:   3
          N Strips:   0
           X Bounds:   0.000e+00, 1.000e+01
          Y Bounds:   0.000e+00, 2.500e+01
          Z Bounds:   -1.270e+02, -1.250e+02
          N Arrays:   0

        """
        cells = [len(self.point_ids)] + list(range(len(self.point_ids)))
        if self.dimension == 0:
            return pyvista.PolyData(self.points.copy(), verts=cells)
        if self.dimension == 1:
            return pyvista.PolyData(self.points.copy(), lines=cells)
        if self.dimension == 2:
            if self.type == CellType.TRIANGLE_STRIP:
                return pyvista.PolyData(self.points.copy(), strips=cells)
            else:
                return pyvista.PolyData(self.points.copy(), faces=cells)
        else:
            raise ValueError(f"3D cells cannot be cast to PolyData: got cell type {self.type}")

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
        if self.type == CellType.POLYHEDRON:
            # construct from faces
            cell_ids = [self.n_faces]
            for face in self.faces:
                cell_ids.append(len(face.point_ids))
                cell_ids.extend(self.point_ids.index(i) for i in face.point_ids)
            cell_ids.insert(0, len(cell_ids))
        else:
            cell_ids = [len(self.point_ids)] + list(range(len(self.point_ids)))
        return pyvista.UnstructuredGrid(
            cell_ids,
            [int(self.type)],
            self.points.copy(),
        )

    @property
    def dimension(self) -> int:  # numpydoc ignore=RT01
        """Return the cell dimension.

        This returns the dimensionality of the cell. For example, 1 for an edge,
        2 for a triangle, and 3 for a tetrahedron.

        Returns
        -------
        int
            The cell dimension.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.get_cell(0).dimension
        2
        """
        return self.GetCellDimension()

    @property
    def n_points(self) -> int:  # numpydoc ignore=RT01
        """Get the number of points composing the cell.

        Returns
        -------
        int
            The number of points.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.get_cell(0).n_points
        3
        """
        return self.GetNumberOfPoints()

    @property
    def n_faces(self) -> int:  # numpydoc ignore=RT01
        """Get the number of faces composing the cell.

        Returns
        -------
        int
            The number of faces.

        Examples
        --------
        >>> from pyvista.examples.cells import Tetrahedron
        >>> mesh = Tetrahedron()
        >>> mesh.get_cell(0).n_faces
        4
        """
        return self.GetNumberOfFaces()

    @property
    def n_edges(self) -> int:  # numpydoc ignore=RT01
        """Get the number of edges composing the cell.

        Returns
        -------
        int
            The number of edges composing the cell.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.get_cell(0).n_edges
        3
        """
        return self.GetNumberOfEdges()

    @property
    def point_ids(self) -> List[int]:  # numpydoc ignore=RT01
        """Get the point IDs composing the cell.

        Returns
        -------
        List[int]
            The point IDs composing the cell.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.get_cell(0).point_ids
        [2, 30, 0]
        """
        point_ids = self.GetPointIds()
        return [point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds())]

    @property
    def points(self) -> np.ndarray:  # numpydoc ignore=RT01
        """Get the point coordinates of the cell.

        Returns
        -------
        np.ndarray
            The point coordinates of the cell.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.get_cell(0).points
        array([[0.05405951, 0.        , 0.49706897],
               [0.05287818, 0.0112396 , 0.49706897],
               [0.        , 0.        , 0.5       ]])
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

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
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
    def edges(self) -> List[Cell]:  # numpydoc ignore=RT01
        """Return a list of edges composing the cell.

        Returns
        -------
        List[Cell]
            A list of edges composing the cell.

        Examples
        --------
        >>> from pyvista.examples.cells import Hexahedron
        >>> mesh = Hexahedron()
        >>> cell = mesh.get_cell(0)
        >>> edges = cell.edges
        >>> len(edges)
        12

        """
        return [self.get_edge(i) for i in range(self.n_edges)]

    @property
    def faces(self) -> List[Cell]:  # numpydoc ignore=RT01
        """Return a list of faces composing the cell.

        Returns
        -------
        List[Cell]
            A list of faces composing the cell.

        Examples
        --------
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
    def bounds(self) -> Tuple[float, float, float, float, float, float]:  # numpydoc ignore=RT01
        """Get the cell bounds in ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            The cell bounds in ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.get_cell(0).bounds
        (0.0, 0.05405950918793678, 0.0, 0.011239604093134403, 0.49706897139549255, 0.5)

        """
        return self.GetBounds()

    @property
    def center(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
        """Get the center of the cell.

        Uses parametric coordinate center to determine x-y-z center.

        Returns
        -------
        Tuple[float, float, float]
            The center of the cell.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.get_cell(0).center
        (0.03564589594801267, 0.0037465346977114677, 0.49804598093032837)

        """
        para_center = [0.0, 0.0, 0.0]
        sub_id = self.GetParametricCenter(para_center)
        # EvaluateLocation requires mutable sub_id
        sub_id = _vtk.mutable(sub_id)
        # center and weights are returned from EvaluateLocation
        center = [0.0, 0.0, 0.0]
        weights = [0.0] * self.n_points
        self.EvaluateLocation(sub_id, para_center, center, weights)
        return cast(Tuple[float, float, float], tuple(center))

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


class CellArray(_vtk.vtkCellArray):
    """PyVista wrapping of vtkCellArray.

    Provides convenience functions to simplify creating a CellArray from
    a numpy array or list.

    Parameters
    ----------
    cells : np.ndarray or list, optional
        Import an array of data with the legacy vtkCellArray layout, e.g.
        ``{ n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ... }``
        Where n0 is the number of points in cell 0, and pX_Y is the Y'th
        point in cell X.

    n_cells : int, optional
        The number of cells.

    deep : bool, default: False
        Perform a deep copy of the original cell.

    Examples
    --------
    Create a cell array containing two triangles from the traditional interleaved format

    >>> from pyvista.core.cell import CellArray
    >>> cellarr = CellArray([3, 0, 1, 2, 3, 3, 4, 5])

    Create a cell array containing two triangles from separate offsets and connectivity arrays

    >>> from pyvista.core.cell import CellArray
    >>> offsets = [0, 3, 6]
    >>> connectivity = [0, 1, 2, 3, 4, 5]
    >>> cellarr = CellArray.from_arrays(offsets, connectivity)
    """

    def __init__(
        self,
        cells: Optional[Union[IntMatrix, IntVector]] = None,
        n_cells: Optional[int] = None,
        deep: bool = False,
    ):
        """Initialize a vtkCellArray."""
        self.__offsets: Optional[_vtk.vtkIdTypeArray] = None
        self.__connectivity: Optional[_vtk.vtkIdTypeArray] = None
        if cells is not None:
            self._set_cells(np.asarray(cells), n_cells, deep)

    def _set_cells(self, cells: NumpyIntArray, n_cells: Optional[int], deep: bool) -> None:
        """Set a vtkCellArray."""
        vtk_idarr, cells = numpy_to_idarr(cells, deep=deep, return_ind=True)  # type: ignore

        # Get number of cells if None.  This is quite a performance
        # bottleneck and we can consider adding a warning.  Good
        # candidate for Cython or JIT compilation
        if n_cells is None:
            if cells.ndim == 1:
                n_cells = ncells_from_cells(cells)
            else:
                n_cells = cells.shape[0]

        self.SetCells(n_cells, vtk_idarr)
        self.__offsets = self.__connectivity = None
        return None

    @property
    def cells(self) -> np.ndarray:  # numpydoc ignore=RT01
        """Return a numpy array of the cells.

        Returns
        -------
        np.ndarray
            A numpy array of the cells.
        """
        return _vtk.vtk_to_numpy(self.GetData()).ravel()

    @property
    def n_cells(self) -> int:  # numpydoc ignore=RT01
        """Return the number of cells.

        Returns
        -------
        int
            The number of cells.
        """
        return self.GetNumberOfCells()

    @property
    def connectivity_array(self) -> NumpyIntArray:  # numpydoc ignore=RT01
        """Return the array with the point ids that define the cells' connectivity.

        Returns
        -------
        np.ndarray
            Array with the point ids that define the cells' connectivity.
        """
        return _get_connectivity_array(self)

    @property
    def offset_array(self) -> NumpyIntArray:  # numpydoc ignore=RT01
        """Return the array used to store cell offsets.

        Returns
        -------
        np.ndarray
            Array used to store cell offsets.
        """
        return _get_offset_array(self)

    def _set_data(self, offsets: IntMatrix, connectivity: IntMatrix, deep: bool = False) -> None:
        """Set the offsets and connectivity arrays."""
        vtk_offsets = cast(_vtk.vtkIdTypeArray, numpy_to_idarr(offsets, deep=deep))
        vtk_connectivity = cast(_vtk.vtkIdTypeArray, numpy_to_idarr(connectivity, deep=deep))
        self.SetData(vtk_offsets, vtk_connectivity)

        # Because vtkCellArray doesn't take ownership of the arrays, it's possible for them to get
        # garbage collected. Keep a reference to them for safety
        self.__offsets = vtk_offsets
        self.__connectivity = vtk_connectivity
        return None

    @staticmethod
    def from_arrays(
        offsets: IntMatrix,
        connectivity: IntMatrix,
        deep: bool = False,
    ) -> CellArray:
        """Construct a CellArray from offsets and connectivity arrays.

        Parameters
        ----------
        offsets : IntMatrix
            Offsets array of length `n_cells + 1`.

        connectivity : IntMatrix
            Connectivity array.

        deep : bool, default: False
            Whether to deep copy the array data into the vtk arrays.

        Returns
        -------
        CellArray
            Constructed CellArray.

        """
        cellarr = CellArray()
        cellarr._set_data(offsets, connectivity, deep=deep)
        return cellarr

    @property
    def regular_cells(self) -> NumpyIntArray:  # numpydoc ignore=RT01
        """Return an array of shape (n_cells, cell_size) of point indices when all faces have the same size.

        Returns
        -------
        numpy.ndarray
            Array of face indices of shape (n_cells, cell_size).

        Notes
        -----
        This property does not validate that the cells are all
        actually the same size. If they're not, this property may either
        raise a `ValueError` or silently return an incorrect array.
        """
        return _get_regular_cells(self)

    @classmethod
    def from_regular_cells(cls, cells: IntMatrix, deep: bool = False) -> pyvista.CellArray:
        """Construct a ``CellArray`` from a (n_cells, cell_size) array of cell indices.

        Parameters
        ----------
        cells : numpy.ndarray or list[list[int]]
            Cell array of shape (n_cells, cell_size) where all cells have the same size `cell_size`.

        deep : bool, default: False
            Whether to deep copy the cell array data into the vtk connectivity array.

        Returns
        -------
        pyvista.CellArray
            Constructed ``CellArray``.
        """
        cells = np.asarray(cells, dtype=pyvista.ID_TYPE)
        n_cells, cell_size = cells.shape
        offsets = cell_size * np.arange(n_cells + 1, dtype=pyvista.ID_TYPE)
        cellarr = cls()
        cellarr._set_data(offsets, cells, deep=deep)
        return cellarr


# The following methods would be much nicer bound to CellArray,
# but then they wouldn't be available on bare vtkCellArrays. In the future,
# consider using vtkCellArray.override decorator, so they're all automatically
# returned as CellArrays


def _get_connectivity_array(cellarr: _vtk.vtkCellArray) -> NumpyIntArray:
    """Return the array with the point ids that define the cells' connectivity."""
    return _vtk.vtk_to_numpy(cellarr.GetConnectivityArray())


def _get_offset_array(cellarr: _vtk.vtkCellArray) -> np.ndarray:
    """Return the array used to store cell offsets."""
    return _vtk.vtk_to_numpy(cellarr.GetOffsetsArray())


def _get_regular_cells(cellarr: _vtk.vtkCellArray) -> NumpyIntArray:
    """Return an array of shape (n_cells, cell_size) of point indices when all faces have the same size."""
    cells = _get_connectivity_array(cellarr)
    if len(cells) == 0:
        return cells

    offsets = _get_offset_array(cellarr)
    cell_size = offsets[1] - offsets[0]
    return cells.reshape(-1, cell_size)
