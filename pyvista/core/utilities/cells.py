"""pyvista wrapping of vtkCellArray."""

from collections import deque
from itertools import count, islice

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk


def ncells_from_cells(cells):
    """Get the number of cells from a VTK cell connectivity array."""
    consumer = deque(maxlen=0)
    it = cells.flat
    for n_cells in count():  # noqa: B007
        skip = next(it, None)
        if skip is None:
            break
        consumer.extend(islice(it, skip))
    return n_cells


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
    vtk_idarr = _vtk.numpy_to_vtkIdTypeArray(ind.ravel(), deep=deep)
    if return_ind:
        return vtk_idarr, ind
    return vtk_idarr


def create_mixed_cells(mixed_cell_dict, nr_points=None):
    """Generate the required cell arrays for the creation of a pyvista.UnstructuredGrid from a cell dictionary.

    This function generates all required cell arrays according to a given cell
    dictionary. The given cell-dictionary should contain a proper
    mapping of vtk_type -> np.ndarray (int), where the given ndarray
    for each cell-type has to be an array of dimensions [N, D] or
    [N*D], where N is the number of cells and D is the size of the
    cells for the given type (e.g. 3 for triangles).  Multiple
    vtk_type keys with associated arrays can be present in one
    dictionary.  This function only accepts cell types of fixed size
    and not dynamic sized cells like ``vtk.VTK_POLYGON``

    Parameters
    ----------
    mixed_cell_dict : dict
        A dictionary that maps VTK-Enum-types (e.g. VTK_TRIANGLE) to
        np.ndarrays of type int.  The ``np.ndarrays`` describe the cell connectivity
    nr_points : int, optional
        Number of points of the grid. Used only to allow additional runtime checks for
        invalid indices, by default None

    Returns
    -------
    cell_types : numpy.ndarray (uint8)
        Types of each cell

    cell_arr : numpy.ndarray (int)
        VTK-cell array

    Raises
    ------
    ValueError
        If any of the cell types are not supported, have dynamic sized
        cells, map to values with wrong size, or cell indices point
        outside the given number of points.

    Examples
    --------
    Create the cell arrays containing two triangles.

    This will generate cell arrays to generate a mesh with two
    disconnected triangles from 6 points.

    >>> import numpy as np
    >>> import vtk
    >>> from pyvista.core.utilities.cells import create_mixed_cells
    >>> cell_types, cell_arr = create_mixed_cells(
    ...     {vtk.VTK_TRIANGLE: np.array([[0, 1, 2], [3, 4, 5]])}
    ... )
    """
    from .cell_type_helper import enum_cell_type_nr_points_map

    if not np.all([k in enum_cell_type_nr_points_map for k in mixed_cell_dict.keys()]):
        raise ValueError("Found unknown or unsupported VTK cell type in your requested cells")

    if not np.all([enum_cell_type_nr_points_map[k] > 0 for k in mixed_cell_dict.keys()]):
        raise ValueError(
            "You requested a cell type with variable length, which can't be used in this method"
        )

    final_cell_types = []
    final_cell_arr = []
    for elem_t, cells_arr in mixed_cell_dict.items():
        nr_points_per_elem = enum_cell_type_nr_points_map[elem_t]
        if (
            not isinstance(cells_arr, np.ndarray)
            or not np.issubdtype(cells_arr.dtype, np.integer)
            or cells_arr.ndim not in [1, 2]
            or (cells_arr.ndim == 1 and cells_arr.size % nr_points_per_elem != 0)
            or (cells_arr.ndim == 2 and cells_arr.shape[-1] != nr_points_per_elem)
        ):
            raise ValueError(
                f"Expected an np.ndarray of size [N, {nr_points_per_elem}] or [N*{nr_points_per_elem}] with an integral type"
            )

        if np.any(cells_arr < 0):
            raise ValueError(f"Non-valid index (<0) given for cells of type {elem_t}")

        if nr_points is not None and np.any(cells_arr >= nr_points):
            raise ValueError(f"Non-valid index (>={nr_points}) given for cells of type {elem_t}")

        if cells_arr.ndim == 1:  # Flattened array present
            cells_arr = cells_arr.reshape([-1, nr_points_per_elem])

        nr_elems = cells_arr.shape[0]
        final_cell_types.append(np.array([elem_t] * nr_elems, dtype=np.uint8))
        final_cell_arr.append(
            np.concatenate(
                [np.ones_like(cells_arr[..., :1]) * nr_points_per_elem, cells_arr], axis=-1
            ).reshape([-1])
        )

    final_cell_types = np.concatenate(final_cell_types)
    final_cell_arr = np.concatenate(final_cell_arr)

    return final_cell_types, final_cell_arr


def get_mixed_cells(vtkobj):
    """Create the cells dictionary from the given pyvista.UnstructuredGrid.

    This functions creates a cells dictionary (see
    create_mixed_cells), with a mapping vtk_type -> np.ndarray (int)
    for fixed size cell types. The returned dictionary will have
    arrays of size [N, D], where N is the number of cells and D is the
    size of the cells for the given type (e.g. 3 for triangles).

    Parameters
    ----------
    vtkobj : pyvista.UnstructuredGrid
        The unstructured grid for which the cells dictionary should be computed

    Returns
    -------
    dict
        Dictionary of cells.

    Raises
    ------
    ValueError
        If vtkobj is not a pyvista.UnstructuredGrid, any of the
        present cells are unsupported, or have dynamic cell sizes,
        like VTK_POLYGON.
    """
    from .cell_type_helper import enum_cell_type_nr_points_map

    return_dict = {}

    if not isinstance(vtkobj, pyvista.UnstructuredGrid):
        raise ValueError("Expected a pyvista object")

    nr_cells = vtkobj.n_cells
    if nr_cells == 0:
        return None

    cell_types = vtkobj.celltypes
    cells = vtkobj.cells

    unique_cell_types = np.unique(cell_types)

    if not np.all([k in enum_cell_type_nr_points_map for k in unique_cell_types]):
        raise ValueError("Found unknown or unsupported VTK cell type in the present cells")

    if not np.all([enum_cell_type_nr_points_map[k] > 0 for k in unique_cell_types]):
        raise ValueError(
            "You requested a cell-dictionary with a variable length cell, which is not supported "
            "currently"
        )

    cell_sizes = np.zeros_like(cell_types)
    for cell_type in unique_cell_types:
        mask = cell_types == cell_type
        cell_sizes[mask] = enum_cell_type_nr_points_map[cell_type]

    cell_ends = np.cumsum(cell_sizes + 1)
    cell_starts = np.concatenate([np.array([0], dtype=cell_ends.dtype), cell_ends[:-1]]) + 1

    for cell_type in unique_cell_types:
        cell_size = enum_cell_type_nr_points_map[cell_type]
        mask = cell_types == cell_type
        current_cell_starts = cell_starts[mask]

        cells_inds = current_cell_starts[..., np.newaxis] + np.arange(cell_size)[np.newaxis].astype(
            cell_starts.dtype
        )

        return_dict[cell_type] = cells[cells_inds]

    return return_dict
