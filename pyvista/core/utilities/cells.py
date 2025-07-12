"""PyVista wrapping of :vtk:`vtkCellArray`."""

from __future__ import annotations

from collections import deque
from itertools import count
from itertools import islice
from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk

if TYPE_CHECKING:
    from pyvista import UnstructuredGrid
    from pyvista.core._typing_core import ArrayLike
    from pyvista.core._typing_core import NumpyArray


def ncells_from_cells(cells: NumpyArray[int]) -> int:
    """Get the number of cells from a VTK cell connectivity array.

    Parameters
    ----------
    cells : numpy.ndarray
        A VTK cell connectivity array.

    Returns
    -------
    int
        The number of cells extracted from the given cell connectivity array.

    """
    consumer: deque[NumpyArray[int]] = deque(maxlen=0)
    it = cells.flat
    for n_cells in count():  # noqa: B007
        skip = next(it, None)
        if skip is None:
            break
        consumer.extend(islice(it, skip))  # type: ignore[arg-type]
    return n_cells


@overload
def numpy_to_idarr(
    ind: int | ArrayLike[int],
    deep: bool = ...,  # noqa: FBT001
    return_ind: Literal[True] = True,  # noqa: FBT002
) -> _vtk.vtkIdTypeArray: ...
@overload
def numpy_to_idarr(
    ind: int | ArrayLike[int],
    deep: bool = ...,  # noqa: FBT001
    return_ind: Literal[False] = False,  # noqa: FBT002
) -> tuple[_vtk.vtkIdTypeArray, NumpyArray[int]]: ...
@overload
def numpy_to_idarr(
    ind: int | ArrayLike[int],
    deep: bool = ...,  # noqa: FBT001
    return_ind: bool = ...,  # noqa: FBT001
) -> tuple[_vtk.vtkIdTypeArray, NumpyArray[int]] | _vtk.vtkIdTypeArray: ...
@_deprecate_positional_args(allowed=['ind'])
def numpy_to_idarr(
    ind: int | ArrayLike[int],
    deep: bool = False,  # noqa: FBT001, FBT002
    return_ind: bool = False,  # noqa: FBT001, FBT002
) -> tuple[_vtk.vtkIdTypeArray, NumpyArray[int]] | _vtk.vtkIdTypeArray:
    """Safely convert a numpy array to a :vtk:`vtkIdTypeArray`.

    Parameters
    ----------
    ind : sequence[int]
        Input sequence to be converted to a :vtk:`vtkIdTypeArray`. Can be either a mask
        or an integer array-like.
    deep : bool, default: False
        If ``True``, deep copy the input data. If ``False``, do not deep copy
        the input data.
    return_ind : bool, default: False
        If ``True``, also return the input array after it has been cast to the
        proper dtype.

    Returns
    -------
    :vtk:`vtkIdTypeArray`
        Converted array as a :vtk:`vtkIdTypeArray`.
    numpy.ndarray
        The input array after it has been cast to the proper dtype. Only
        returned if `return_ind` is set to ``True``.

    Raises
    ------
    TypeError
        If the input array is not a mask or an integer array-like.

    """
    ind = np.asarray(ind)

    # np.asarray will eat anything, so we have to weed out bogus inputs
    if not issubclass(ind.dtype.type, (np.bool_, np.integer)):
        msg = 'Indices must be either a mask or an integer array-like'
        raise TypeError(msg)

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


def create_mixed_cells(
    mixed_cell_dict: dict[np.uint8, NumpyArray[int]], nr_points: int | None = None
) -> tuple[NumpyArray[np.uint8], NumpyArray[int]]:
    """Generate cell arrays for the creation of a pyvista.UnstructuredGrid from a cell dictionary.

    This function generates all required cell arrays according to a given cell
    dictionary. The given cell-dictionary should contain a proper
    mapping of vtk_type -> np.ndarray (int), where the given ndarray
    for each cell-type has to be an array of dimensions [N, D] or
    [N*D], where N is the number of cells and D is the size of the
    cells for the given type (e.g. 3 for triangles).  Multiple
    vtk_type keys with associated arrays can be present in one
    dictionary.  This function only accepts cell types of fixed size
    and not dynamic sized cells like :attr:`~pyvista.CellType.POLYGON`

    Parameters
    ----------
    mixed_cell_dict : dict
        A dictionary that maps VTK-Enum-types (e.g. :attr:`~pyvista.CellType.TRIANGLE`) to
        np.ndarrays of type int.  The ``np.ndarrays`` describe the cell
        connectivity.
    nr_points : int, optional
        Number of points of the grid. Used only to allow additional runtime
        checks for invalid indices.

    Returns
    -------
    cell_types : numpy.ndarray (uint8)
        Types of each cell.

    cell_arr : numpy.ndarray (int)
        VTK-cell array.

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
    from .cell_type_helper import enum_cell_type_nr_points_map  # noqa: PLC0415

    if not np.all([k in enum_cell_type_nr_points_map for k in mixed_cell_dict.keys()]):
        msg = 'Found unknown or unsupported VTK cell type in your requested cells'
        raise ValueError(msg)

    if not np.all([enum_cell_type_nr_points_map[k] > 0 for k in mixed_cell_dict.keys()]):
        msg = "You requested a cell type with variable length, which can't be used in this method"
        raise ValueError(msg)

    final_cell_types = []
    final_cell_arr = []
    for elem_t, cells_arr in mixed_cell_dict.items():
        nr_points_per_elem = enum_cell_type_nr_points_map[elem_t]
        if (
            not isinstance(cells_arr, np.ndarray)  # type: ignore[redundant-expr]
            or not np.issubdtype(cells_arr.dtype, np.integer)
            or cells_arr.ndim not in [1, 2]
            or (cells_arr.ndim == 1 and cells_arr.size % nr_points_per_elem != 0)
            or (cells_arr.ndim == 2 and cells_arr.shape[-1] != nr_points_per_elem)
        ):
            msg = (
                f'Expected an np.ndarray of size [N, {nr_points_per_elem}] or '
                f'[N*{nr_points_per_elem}] with an integral type'
            )
            raise ValueError(msg)

        if np.any(cells_arr < 0):
            msg = f'Non-valid index (<0) given for cells of type {elem_t}'
            raise ValueError(msg)

        if nr_points is not None and np.any(cells_arr >= nr_points):
            msg = f'Non-valid index (>={nr_points}) given for cells of type {elem_t}'
            raise ValueError(msg)

        # Ensure array is not flat
        cells_arr_not_flat = (
            cells_arr.reshape([-1, nr_points_per_elem]) if cells_arr.ndim == 1 else cells_arr
        )

        nr_elems = cells_arr_not_flat.shape[0]
        final_cell_types.append(np.array([elem_t] * nr_elems, dtype=np.uint8))
        final_cell_arr.append(
            np.concatenate(
                [
                    np.ones_like(cells_arr_not_flat[..., :1]) * nr_points_per_elem,
                    cells_arr_not_flat,
                ],
                axis=-1,
            ).reshape([-1]),
        )

    cell_types_out = np.concatenate(final_cell_types)
    cell_arr_out = np.concatenate(final_cell_arr)

    return cell_types_out, cell_arr_out


def get_mixed_cells(vtkobj: UnstructuredGrid) -> dict[np.uint8, NumpyArray[int]]:
    """Create the cells dictionary from the given pyvista.UnstructuredGrid.

    This functions creates a cells dictionary (see
    create_mixed_cells), with a mapping vtk_type -> np.ndarray (int)
    for fixed size cell types. The returned dictionary will have
    arrays of size [N, D], where N is the number of cells and D is the
    size of the cells for the given type (e.g. 3 for triangles).

    .. versionchanged:: 0.46

        An empty dict ``{}`` is returned instead of ``None`` if the input
        is empty.

    Parameters
    ----------
    vtkobj : pyvista.UnstructuredGrid
        The unstructured grid for which the cells dictionary should be computed.

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
    from .cell_type_helper import enum_cell_type_nr_points_map  # noqa: PLC0415

    return_dict: dict[np.uint8, NumpyArray[int]] = {}

    if not isinstance(vtkobj, pyvista.UnstructuredGrid):
        msg = 'Expected a pyvista object'  # type: ignore[unreachable]
        raise TypeError(msg)

    nr_cells = vtkobj.n_cells
    if nr_cells == 0:
        return return_dict

    cell_types = vtkobj.celltypes
    cells = vtkobj.cells

    unique_cell_types = np.unique(cell_types)

    if not np.all([k in enum_cell_type_nr_points_map for k in unique_cell_types]):
        msg = 'Found unknown or unsupported VTK cell type in the present cells'
        raise ValueError(msg)

    if not np.all([enum_cell_type_nr_points_map[k] > 0 for k in unique_cell_types]):
        msg = (
            'You requested a cell-dictionary with a variable length cell, which is not supported '
            'currently'
        )
        raise ValueError(msg)

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

        cells_inds = current_cell_starts[..., np.newaxis] + np.arange(cell_size)[
            np.newaxis
        ].astype(
            cell_starts.dtype,
        )

        return_dict[cell_type] = cells[cells_inds]

    return return_dict
