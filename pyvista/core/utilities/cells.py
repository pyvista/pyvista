"""PyVista wrapping of :vtk:`vtkCellArray`."""

from __future__ import annotations

from collections import deque
from itertools import count
from itertools import islice
from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista._deprecate_positional_args import _deprecate_positional_args

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista import CellType
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
        Input sequence to be converted to a :vtk:`vtkIdTypeArray`. Can be
        either a mask or an integer array-like.
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
    if not (np.issubdtype(ind.dtype, np.integer) or ind.dtype == np.bool_):
        msg = 'Indices must be either a mask or an integer array-like'
        raise TypeError(msg)

    if ind.dtype == np.bool_:
        ind = ind.nonzero()[0].astype(pv.ID_TYPE)
    elif ind.dtype != pv.ID_TYPE:
        ind = ind.astype(pv.ID_TYPE)
    elif not ind.flags['C_CONTIGUOUS']:
        ind = np.ascontiguousarray(ind, dtype=pv.ID_TYPE)

    # must ravel or segfault when saving MultiBlock
    # but skip the ``ravel()`` allocation when the array is already
    # 1D and contiguous (the common case), since ndarray.ravel() of
    # a non-1D shape returns a copy.
    ravelled = ind if ind.ndim == 1 else ind.ravel()
    vtk_idarr = _vtk.numpy_to_vtkIdTypeArray(ravelled, deep=deep)
    if return_ind:
        return vtk_idarr, ind
    return vtk_idarr


def _cell_type_n_points(cell_type: CellType) -> int | None:
    """Return the fixed number of points for a cell type, or ``None`` if data-defined.

    Composite, higher-order, polygonal and polyhedral cells do not have a fixed
    number of points (:attr:`~pyvista.CellType.n_points` raises for them), so the
    per-cell point count has to come from the connectivity data instead.
    """
    try:
        n_points = cell_type.n_points
    except ValueError:
        return None
    return n_points if n_points > 0 else None


def _check_cell_indices(indices: NumpyArray[int], elem_t: CellType, nr_points: int | None) -> None:
    """Validate that connectivity indices are non-negative and within the point count."""
    if np.any(indices < 0):
        msg = f'Non-valid index (<0) given for cells of type {elem_t}'
        raise ValueError(msg)
    if nr_points is not None and np.any(indices >= nr_points):
        msg = f'Non-valid index (>={nr_points}) given for cells of type {elem_t}'
        raise ValueError(msg)


def _fixed_size_cells(
    elem_t: CellType,
    nr_points_per_elem: int,
    cells_arr: NumpyArray[int],
    *,
    nr_points: int | None,
) -> tuple[NumpyArray[np.uint8], NumpyArray[int]]:
    """Build the cell-type and connectivity arrays for a fixed-size cell type."""
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

    _check_cell_indices(cells_arr, elem_t, nr_points)

    # Ensure array is not flat
    not_flat = cells_arr.reshape([-1, nr_points_per_elem]) if cells_arr.ndim == 1 else cells_arr
    nr_elems = not_flat.shape[0]
    types = np.array([elem_t] * nr_elems, dtype=np.uint8)
    arr = np.concatenate(
        [np.ones_like(not_flat[..., :1]) * nr_points_per_elem, not_flat],
        axis=-1,
    ).reshape([-1])
    return types, arr


def _variable_size_cells(
    elem_t: CellType,
    cells_arr: NumpyArray[int] | Sequence[ArrayLike[int]],
    *,
    nr_points: int | None,
) -> tuple[NumpyArray[np.uint8], NumpyArray[int]]:
    """Build the cell-type and connectivity arrays for a data-defined cell type.

    The per-cell point count is taken from the data: a 2D ``[N, D]`` array maps to
    ``N`` cells of ``D`` points each, while a sequence of 1D integer arrays maps to
    cells of differing sizes (one array per cell).
    """
    if elem_t == pv.CellType.POLYHEDRON:
        msg = (
            "Cell type 'POLYHEDRON' cannot be created from a cells dict because a "
            'polyhedron is defined by its faces, not a flat list of point indices. '
            'Build the UnstructuredGrid from explicit cell and face arrays instead.'
        )
        raise ValueError(msg)

    # Uniform case: a 2D [N, D] array is N cells with D points each.
    if isinstance(cells_arr, np.ndarray):
        if cells_arr.ndim == 2 and np.issubdtype(cells_arr.dtype, np.integer):
            _check_cell_indices(cells_arr, elem_t, nr_points)
            nr_elems, nr_points_per_elem = cells_arr.shape
            types = np.array([elem_t] * nr_elems, dtype=np.uint8)
            counts = np.ones_like(cells_arr[..., :1]) * nr_points_per_elem
            arr = np.concatenate([counts, cells_arr], axis=-1).reshape([-1])
            return types, arr
        msg = (
            f"Cell type '{elem_t.name}' has a data-defined number of points. Pass a "
            f'2D [N, D] array for cells that all have D points, or a sequence of 1D '
            f'integer arrays for cells of differing sizes (a flat 1D array is ambiguous).'
        )
        raise ValueError(msg)

    # Ragged case: a sequence of per-cell 1D index arrays.
    per_cell = [np.asarray(cell) for cell in cells_arr]
    chunks = []
    for cell in per_cell:
        if cell.ndim != 1 or not np.issubdtype(cell.dtype, np.integer) or cell.size == 0:
            msg = (
                f"Each cell of type '{elem_t.name}' must be a non-empty 1D array of "
                f'integer point indices.'
            )
            raise ValueError(msg)
        _check_cell_indices(cell, elem_t, nr_points)
        chunks.append(np.concatenate([[cell.size], cell]).astype(np.int64))
    types = np.array([elem_t] * len(per_cell), dtype=np.uint8)
    arr = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    return types, arr


def create_mixed_cells(
    mixed_cell_dict: dict[np.uint8, NumpyArray[int] | Sequence[ArrayLike[int]]],
    nr_points: int | None = None,
) -> tuple[NumpyArray[np.uint8], NumpyArray[int]]:
    """Generate cell arrays for the creation of a pyvista.UnstructuredGrid from a cell dictionary.

    This function generates all required cell arrays according to a given cell
    dictionary. The given cell-dictionary should contain a proper
    mapping of vtk_type -> np.ndarray (int), where the given ndarray
    for each cell-type has to be an array of dimensions [N, D] or
    [N*D], where N is the number of cells and D is the size of the
    cells for the given type (e.g. 3 for triangles).  Multiple
    vtk_type keys with associated arrays can be present in one
    dictionary.

    Cell types whose number of points is not fixed (e.g.
    :attr:`~pyvista.CellType.POLYGON`, :attr:`~pyvista.CellType.POLY_VERTEX`, and
    the higher-order :attr:`~pyvista.CellType.LAGRANGE_TRIANGLE` /
    :attr:`~pyvista.CellType.BEZIER_TRIANGLE` families) are also supported. For
    such a type, pass either a 2D ``[N, D]`` array (``N`` cells that all have ``D``
    points) or, when the cells differ in size, a sequence of 1D integer arrays (one
    array of point indices per cell). :attr:`~pyvista.CellType.POLYHEDRON` is the
    one exception: it is defined by its faces rather than a flat point list and so
    cannot be created from a cells dict.

    .. versionchanged:: 0.49

        Cell types with a data-defined number of points are now supported.

    Parameters
    ----------
    mixed_cell_dict : dict
        A dictionary that maps VTK-Enum-types (e.g. :attr:`~pyvista.CellType.TRIANGLE`) to
        np.ndarrays of type int.  The ``np.ndarrays`` describe the cell
        connectivity. For cell types with a data-defined number of points, the value
        may instead be a sequence of 1D integer arrays (one per cell).
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
        If any of the cell types are not supported, map to values with the
        wrong size, or cell indices point outside the given number of points.

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

    Create the cell arrays for two polygons of differing size (a triangle and a
    quad) by passing a sequence of one index array per cell.

    >>> import pyvista as pv
    >>> cell_types, cell_arr = create_mixed_cells(
    ...     {pv.CellType.POLYGON: [np.array([0, 1, 2]), np.array([3, 4, 5, 6])]}
    ... )

    """
    final_cell_types = []
    final_cell_arr = []
    for key, cells_arr in mixed_cell_dict.items():
        elem_t = pv.CellType(key)  # type: ignore[arg-type]
        nr_points_per_elem = _cell_type_n_points(elem_t)
        if nr_points_per_elem is not None:
            types, arr = _fixed_size_cells(
                elem_t,
                nr_points_per_elem,
                cells_arr,  # type: ignore[arg-type]
                nr_points=nr_points,
            )
        else:
            types, arr = _variable_size_cells(elem_t, cells_arr, nr_points=nr_points)
        final_cell_types.append(types)
        final_cell_arr.append(arr)

    cell_types_out = np.concatenate(final_cell_types)
    cell_arr_out = np.concatenate(final_cell_arr)

    return cell_types_out, cell_arr_out


def get_mixed_cells(
    vtkobj: UnstructuredGrid,
) -> dict[np.uint8, NumpyArray[int] | list[NumpyArray[int]]]:
    """Create the cells dictionary from the given pyvista.UnstructuredGrid.

    This functions creates a cells dictionary (see
    create_mixed_cells), with a mapping vtk_type -> np.ndarray (int).
    For a cell type whose cells all have the same number of points, the
    value is an array of size [N, D], where N is the number of cells and
    D is the size of the cells for the given type (e.g. 3 for triangles).
    For a cell type with a data-defined number of points whose cells differ
    in size (e.g. :attr:`~pyvista.CellType.POLYGON`), the value is instead a
    list of N 1D arrays, one per cell. Both forms round-trip through
    :func:`create_mixed_cells`.

    .. versionchanged:: 0.46

        An empty dict ``{}`` is returned instead of ``None`` if the input
        is empty.

    .. versionchanged:: 0.49

        Cell types with a data-defined number of points are now supported.

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
        If vtkobj is not a pyvista.UnstructuredGrid, any of the present
        cells are unsupported, or any cell is a
        :attr:`~pyvista.CellType.POLYHEDRON` (which is defined by its faces
        and cannot be represented as a flat point list).

    """
    if not isinstance(vtkobj, pv.UnstructuredGrid):
        msg = 'Expected a pyvista object'  # type: ignore[unreachable]
        raise TypeError(msg)

    nr_cells = vtkobj.n_cells
    if nr_cells == 0:
        return {}

    cell_types = vtkobj.celltypes
    connectivity = vtkobj.cell_connectivity
    offset = vtkobj.offset

    # Derive the distinct cell types from the live ``celltypes`` array rather than
    # ``vtkobj.distinct_cell_types`` (which VTK may cache, going stale after a raw
    # mutation). Building each ``CellType`` here also validates that every present
    # type is known.
    distinct_cell_types = [pv.CellType(int(t)) for t in np.unique(cell_types)]

    if pv.CellType.POLYHEDRON in distinct_cell_types:
        msg = (
            "Cell type 'POLYHEDRON' cannot be represented in a cells dict because a "
            'polyhedron is defined by its faces, not a flat list of point indices.'
        )
        raise ValueError(msg)

    # Per-cell point counts come from the offset array, so this works uniformly for
    # both fixed-size and data-defined (variable) cell types.
    cell_sizes = np.diff(offset)
    cell_starts = offset[:-1]

    return_dict: dict[np.uint8, NumpyArray[int] | list[NumpyArray[int]]] = {}
    for cell_type in distinct_cell_types:
        mask = cell_types == cell_type
        starts = cell_starts[mask]
        sizes = cell_sizes[mask]

        if np.all(sizes == sizes[0]):
            cell_size = int(sizes[0])
            cells_inds = starts[..., np.newaxis] + np.arange(cell_size, dtype=starts.dtype)
            return_dict[np.uint8(cell_type)] = connectivity[cells_inds]
        else:
            return_dict[np.uint8(cell_type)] = [
                connectivity[start : start + size]
                for start, size in zip(starts, sizes, strict=True)
            ]

    return return_dict
