"""pyvista wrapping of vtkCellArray."""
import numpy as np
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtk import vtkCellArray
import vtk

import pyvista
from .cell_type_helpers.cell_selector import enum_cell_type_nr_points_map
VTK9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9

def numpy_to_idarr(ind, deep=False, return_ind=False):
    """Safely convert a numpy array to a vtkIdTypeArray."""
    ind = np.asarray(ind)

    # np.asarray will eat anything, so we have to weed out bogus inputs
    if not issubclass(ind.dtype.type, (np.bool_, np.integer)):
        raise TypeError('Indices must be either a mask or an integer array-like')

    if ind.dtype == np.bool_:
        ind = ind.nonzero()[0].astype(pyvista.ID_TYPE)
    elif ind.dtype != pyvista.ID_TYPE:
        ind = ind.astype(pyvista.ID_TYPE)
    elif not ind.flags['C_CONTIGUOUS']:
        ind = np.ascontiguousarray(ind, dtype=pyvista.ID_TYPE)

    # must ravel or segfault when saving MultiBlock
    vtk_idarr = numpy_to_vtkIdTypeArray(ind.ravel(), deep=deep)
    if return_ind:
        return vtk_idarr, ind
    return vtk_idarr


class CellArray(vtkCellArray):
    """pyvista wrapping of vtkCellArray.

    Provides convenience functions to simplify creating a CellArray from
    a numpy array or list.

    Import an array of data with the legacy vtkCellArray layout, e.g.

    ``{ n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ... }``
    Where n0 is the number of points in cell 0, and pX_Y is the Y'th
    point in cell X.

    Examples
    --------
    Create a cell array containing two triangles.

    >>> from pyvista.utilities.cells import CellArray
    >>> cellarr = CellArray([3, 0, 1, 2, 3, 3, 4, 5])
    """

    def __init__(self, cells=None, n_cells=None, deep=False):
        """Initialize a vtkCellArray."""
        if cells is not None:
            self._set_cells(cells, n_cells, deep)

    def _set_cells(self, cells, n_cells, deep):
        vtk_idarr, cells = numpy_to_idarr(cells, deep=deep, return_ind=True)
        # get number of cells if none
        if n_cells is None:
            if cells.ndim == 1:
                c = 0
                n_cells = 0
                while c < cells.size:
                    c += cells[c] + 1
                    n_cells += 1
            else:
                n_cells = cells.shape[0]

        self.SetCells(n_cells, vtk_idarr)

    @property
    def cells(self):
        """Return a numpy array of the cells."""
        return vtk_to_numpy(self.GetData()).ravel()

    @property
    def n_cells(self):
        """Return the number of cells."""
        return self.GetNumberOfCells()

def generate_cell_offsets_loop(cells, cell_types):
    offsets = np.zeros(shape=[cell_types.size], dtype=np.int64)

    current_cell_pos = 0
    for cell_i, cell_t in enumerate(cell_types):
        if current_cell_pos >= cell_types.size:
            raise ValueError("Cell types and cell array are inconsistent. Got %d values left after reading all types" % (cell_types.size - current_cell_pos))

        cell_size = cells[current_cell_pos]
        offsets[cell_i] = current_cell_pos
        current_cell_pos += cell_size+1

    if current_cell_pos != cell_types.size:
        raise ValueError("Cell types and cell array are inconsistent. Got %d values left after reading all types" % (cell_types.size - current_cell_pos))

    return offsets


def generate_cell_offsets(cells, cell_types):
    try:
        cell_sizes = np.array([enum_cell_type_nr_points_map[cell_t] for cell_t in cell_types], dtype=np.int32)
    except KeyError as err:
        return generate_cell_offsets_loop(cells, cell_types) #Unknown requested cell type present

    if np.any(cell_sizes == 0):
        return generate_cell_offsets_loop(cells, cell_types)

    cell_sizes_cum = np.cumsum(cell_sizes+1)

    if cell_sizes_cum[-1] != cells.size:
        raise ValueError("Cell types and cell array are inconsistent. Expected a cell array of length %d according to the cell types" % (cell_sizes_cum[-1]))

    offsets = np.concatenate([[0], np.cumsum(cell_sizes_cum)[:-1]])

    return offsets

def create_mixed_cells(mixed_cell_dict):
    if not np.all([k in enum_cell_type_nr_points_map for k in mixed_cell_dict.keys()]):
        raise ValueError("Found unknown or unsupported VTK cell type in your requested cells")

    if not np.all([enum_cell_type_nr_points_map[k] > 0 for k in mixed_cell_dict.keys()]):
        raise ValueError("You requested a cell type with variable length, which can't be used in this method")

    final_cell_types = []
    final_cell_arr = []
    final_cell_offsets = [np.array([0])]
    current_cell_offset = 0
    for elem_t, cells_arr in mixed_cell_dict.items():
        nr_points_per_elem = enum_cell_type_nr_points_map[elem_t]
        if (not isinstance(cells_arr, np.ndarray) or not np.issubdtype(cells_arr.dtype, np.integer)
                or cells_arr.ndim not in [1, 2]
                or (cells_arr.ndim == 1 and cells_arr.size % nr_points_per_elem != 0)
                or (cells_arr.ndim == 2 and cells_arr.shape[-1] != nr_points_per_elem)):
            raise ValueError("Expected an np.ndarray of size [N, %d] or [N*%d] with an integral type" % (nr_points_per_elem, nr_points_per_elem))

        if cells_arr.ndim == 1: #Flattened array present
            cells_arr = cells_arr.reshape([-1, nr_points_per_elem])
            
        nr_elems = cells_arr.shape[0]
        final_cell_types.append(np.array([elem_t] * nr_elems, dtype=np.uint8))
        final_cell_arr.append(np.concatenate([np.ones_like(cells_arr[..., :1]) * nr_points_per_elem, cells_arr], axis=-1).reshape([-1]))

        if not VTK9:
            final_cell_offsets.append(current_cell_offset + (nr_points_per_elem+1) * (np.arange(nr_elems)+1))
            current_cell_offset += final_cell_offsets[-1][-1]

    final_cell_types = np.concatenate(final_cell_types)
    final_cell_arr = np.concatenate(final_cell_arr)

    if not VTK9:
        final_cell_offsets = np.concatenate(final_cell_offsets[:-1])

    if not VTK9:
        return final_cell_types, final_cell_arr, final_cell_offsets
    else:
        return final_cell_types, final_cell_arr

def get_mixed_cells_loop(vtkobj):
    """
    Old fallback method in case variable length elems are present

     Return
    ------
    Returns a dictionary mapping of elements vtk_type (int) -> cells (np.ndarray)
    """
    return_dict = {}

    for i in range(vtkobj.GetNumberOfCells()):
        cell = vtkobj.GetCell(i)
        cell_type = vtkobj.GetCellType(i)
        nr_points_per_elem = cell.GetNumberOfPoints()
        if not cell_type in return_dict:
            return_dict[cell_type] = []

        return_dict[cell_type].append([vtkobj.GetCell(i).GetPointId(j) for j in range(nr_points_per_elem)])

    return return_dict

def get_mixed_cells(vtkobj):
    return_dict = {}

    if not isinstance(vtkobj, pyvista.UnstructuredGrid):
        raise ValueError("Expected a pyvista object")

    nr_cells = vtkobj.n_cells
    if nr_cells == 0:
      return None

    cell_types = vtkobj.celltypes
    cells = vtkobj.cells

    unique_cell_types = np.unique(cell_types)
    cell_sizes = np.zeros_like(cell_types)
    for cell_type in unique_cell_types:
      mask = cell_types == cell_type
      cell_sizes[mask] = enum_cell_type_nr_points_map[cell_type]

    #+1 For the cell sizes
    if np.sum(cell_sizes + 1) != cells.size:
      return get_mixed_cells_loop()

    cell_ends = np.cumsum(cell_sizes + 1)
    cell_starts = np.concatenate([np.array([0], dtype=cell_ends.dtype), cell_ends[:-1]]) + 1

    for cell_type in unique_cell_types:
      cell_size = enum_cell_type_nr_points_map[cell_type]
      mask = cell_types == cell_type
      current_cell_starts = cell_starts[mask]

      cells_inds = current_cell_starts[..., np.newaxis] + np.arange(cell_size)[np.newaxis].astype(cell_starts.dtype)

      return_dict[cell_type] = cells[cells_inds]

    return return_dict
