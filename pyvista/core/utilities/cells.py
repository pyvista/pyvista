"""PyVista wrapping of :vtk:`vtkCellArray`."""

from __future__ import annotations

from collections import deque
import itertools
from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.core._vtk_utilities import _SUPPORTS_FIXED_SIZE_STORAGE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista import CellType
    from pyvista import DataSet
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
    for n_cells in itertools.count():  # noqa: B007
        skip = next(it, None)
        if skip is None:
            break
        consumer.extend(itertools.islice(it, skip))  # type: ignore[arg-type]
    return n_cells


# fmt: off
# ruff: disable[E501]
@overload
def numpy_to_idarr(ind: int | ArrayLike[int], *, deep: bool = ..., return_ind: Literal[False] = False) -> _vtk.vtkIdTypeArray: ...
@overload
def numpy_to_idarr(ind: int | ArrayLike[int], *, deep: bool = ..., return_ind: Literal[True] = ...) -> tuple[_vtk.vtkIdTypeArray, NumpyArray[int]]: ...
@overload
def numpy_to_idarr(ind: int | ArrayLike[int], *, deep: bool = ..., return_ind: bool = ...) -> tuple[_vtk.vtkIdTypeArray, NumpyArray[int]] | _vtk.vtkIdTypeArray: ...
# ruff: enable[E501]
# fmt: on
def numpy_to_idarr(
    ind: int | ArrayLike[int],
    *,
    deep: bool = False,
    return_ind: bool = False,
) -> tuple[_vtk.vtkIdTypeArray, NumpyArray[int]] | _vtk.vtkIdTypeArray:
    """Safely convert a NumPy array to a :vtk:`vtkIdTypeArray`.

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
        proper ``dtype``.

    Returns
    -------
    :vtk:`vtkIdTypeArray`
        Converted array as a :vtk:`vtkIdTypeArray`.
    numpy.ndarray
        The input array after it has been cast to the proper ``dtype``. Only
        returned if ``return_ind`` is set to ``True``.

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

    Composite, higher-order, polygonal, and polyhedral cells do not have a fixed
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
    not_flat = _validate_fixed_size_cells(
        elem_t, nr_points_per_elem, cells_arr, nr_points=nr_points
    )
    nr_elems = not_flat.shape[0]
    types = np.full(nr_elems, elem_t, dtype=np.uint8)
    arr = np.concatenate(
        [np.full_like(not_flat[..., :1], nr_points_per_elem), not_flat],
        axis=-1,
    ).reshape([-1])
    return types, arr


def _validate_fixed_size_cells(
    elem_t: CellType,
    nr_points_per_elem: int,
    cells_arr: NumpyArray[int],
    *,
    nr_points: int | None,
) -> NumpyArray[int]:
    """Validate and reshape connectivity for a fixed-size cell type."""
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
    return cells_arr.reshape([-1, nr_points_per_elem]) if cells_arr.ndim == 1 else cells_arr


def _get_regular_cells_from_dict(
    cells_dict: dict[np.uint8, NumpyArray[int] | Sequence[ArrayLike[int]]],
    nr_points: int,
) -> tuple[NumpyArray[np.uint8], NumpyArray[int]] | None:
    """Return cell types and regular connectivity for a single-type cells dict."""
    if len(cells_dict) != 1:
        return None

    cell_type, cells = next(iter(cells_dict.items()))
    elem_t = pv.CellType(cell_type)  # type: ignore[arg-type]
    if not isinstance(cells, np.ndarray) or elem_t == pv.CellType.POLYHEDRON:
        return None

    nr_points_per_elem = _cell_type_n_points(elem_t)
    if nr_points_per_elem is not None:
        connectivity = _validate_fixed_size_cells(
            elem_t, nr_points_per_elem, cells, nr_points=nr_points
        )
    elif cells.ndim == 2 and np.issubdtype(cells.dtype, np.integer):
        _check_cell_indices(cells, elem_t, nr_points)
        connectivity = cells
    else:
        return None

    cell_types = np.full(connectivity.shape[0], elem_t, dtype=np.uint8)
    return cell_types, connectivity


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
    for cell in per_cell:
        if cell.ndim != 1 or not np.issubdtype(cell.dtype, np.integer) or cell.size == 0:
            msg = (
                f"Each cell of type '{elem_t.name}' must be a non-empty 1D array of "
                f'integer point indices.'
            )
            raise ValueError(msg)
    # Optimization: check the indices and build the array for all cells at once instead
    # of concatenating one cell at a time; each cell's size is inserted ahead of its ids
    sizes = np.array([cell.size for cell in per_cell], dtype=pv.ID_TYPE)
    connectivity: NumpyArray[int] = (
        np.concatenate(per_cell).astype(pv.ID_TYPE, copy=False)
        if per_cell
        else np.empty(0, dtype=pv.ID_TYPE)
    )
    _check_cell_indices(connectivity, elem_t, nr_points)
    types = np.full(len(per_cell), elem_t, dtype=np.uint8)
    arr = np.insert(connectivity, np.cumsum(sizes) - sizes, sizes)
    return types, arr


def create_mixed_cells(
    mixed_cell_dict: dict[np.uint8, NumpyArray[int] | Sequence[ArrayLike[int]]],
    nr_points: int | None = None,
) -> tuple[NumpyArray[np.uint8], NumpyArray[int]]:
    """Generate cell arrays for the creation of a pyvista.UnstructuredGrid from a cell dictionary.

    This function generates all required cell arrays according to a given cell
    dictionary. The given cell-dictionary should contain a proper
    mapping of ``vtk_type`` -> ``np.ndarray`` (int), where the given ``ndarray``
    for each cell-type has to be an array of dimensions [N, D] or
    [N*D], where N is the number of cells and D is the size of the
    cells for the given type (for example, 3 for triangles).  Multiple
    ``vtk_type`` keys with associated arrays can be present in one
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
        A dictionary that maps VTK-Enum-types (for example, :attr:`~pyvista.CellType.TRIANGLE`) to
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
    ``create_mixed_cells``), with a mapping ``vtk_type`` -> ``np.ndarray`` (int).
    For a cell type whose cells all have the same number of points, the
    value is an array of size [N, D], where N is the number of cells and
    D is the size of the cells for the given type (for example, 3 for triangles).
    For a cell type with a data-defined number of points whose cells differ
    in size (for example, :attr:`~pyvista.CellType.POLYGON`), the value is instead a
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
        If ``vtkobj`` is not a pyvista.UnstructuredGrid, any of the present
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
    cell_array = vtkobj.GetCells()
    cell_size = (
        cell_array.IsHomogeneous()
        if _SUPPORTS_FIXED_SIZE_STORAGE and cell_array.IsStorageFixedSize()
        else -1
    )
    regular_connectivity = connectivity.reshape(nr_cells, cell_size) if cell_size >= 0 else None

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

    if regular_connectivity is None:
        offset = vtkobj.cell_offsets
        cell_sizes = np.diff(offset)
        cell_starts = offset[:-1]

    return_dict: dict[np.uint8, NumpyArray[int] | list[NumpyArray[int]]] = {}
    for cell_type in distinct_cell_types:
        mask = cell_types == cell_type
        if regular_connectivity is not None:
            return_dict[np.uint8(cell_type)] = regular_connectivity[mask]
            continue

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


def _cell_edge_lengths(
    mesh: DataSet, cell_ids: NumpyArray[int] | None = None
) -> NumpyArray[float]:
    """Return the length of every edge of the cells of a mesh."""
    if mesh.n_cells == 0:
        return np.empty(0, dtype=float)
    if isinstance(mesh, pv.ImageData):
        return _image_edge_lengths(mesh, cell_ids)
    if isinstance(mesh, pv.RectilinearGrid):
        return _rectilinear_edge_lengths(mesh, cell_ids)
    if isinstance(mesh, pv.StructuredGrid):
        return _structured_edge_lengths(mesh, cell_ids)
    if isinstance(mesh, pv.PolyData):
        return _polydata_edge_lengths(mesh, cell_ids)
    if isinstance(mesh, pv.ExplicitStructuredGrid):
        cell_types = np.full(mesh.n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    else:
        if not isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.cast_to_unstructured_grid()
        cell_types = mesh.celltypes
    offsets = _vtk.vtk_to_numpy(mesh.GetCells().GetOffsetsArray())
    connectivity = _vtk.vtk_to_numpy(mesh.GetCells().GetConnectivityArray())
    return _cell_array_edge_lengths(
        mesh, cell_types=cell_types, offsets=offsets, connectivity=connectivity, cell_ids=cell_ids
    )


def _image_edge_lengths(mesh: pv.ImageData, cell_ids: NumpyArray[int] | None) -> NumpyArray[float]:
    """Return the four edges per axis shared by every cell of an image."""
    spacing = np.array(mesh.spacing, dtype=float)[np.array(mesh.dimensions) > 1]
    n_cells = mesh.n_cells if cell_ids is None else len(cell_ids)
    return np.repeat(spacing, _edges_per_axis(spacing.size) * n_cells)


def _rectilinear_edge_lengths(
    mesh: pv.RectilinearGrid, cell_ids: NumpyArray[int] | None
) -> NumpyArray[float]:
    """Return the four edges per axis of the axis-aligned cells of a rectilinear grid."""
    coords = [np.asarray(c, dtype=float) for c in (mesh.x, mesh.y, mesh.z)]
    steps = [np.diff(c) if c.size > 1 else np.zeros(1) for c in coords]
    cell_ids = np.arange(mesh.n_cells) if cell_ids is None else cell_ids
    k, j, i = np.unravel_index(cell_ids, [step.size for step in steps[::-1]])
    edges = [
        step[index] for step, index, c in zip(steps, (i, j, k), coords, strict=True) if c.size > 1
    ]
    if not edges:
        return np.empty(0)
    return np.repeat(np.concatenate(edges), _edges_per_axis(len(edges)))


def _structured_edge_lengths(
    mesh: pv.StructuredGrid, cell_ids: NumpyArray[int] | None
) -> NumpyArray[float]:
    """Return the edges of the hexahedral cells of a structured grid."""
    dims = np.array(mesh.dimensions)
    points = mesh.points.reshape((*dims[::-1], 3))
    cell_ids = np.arange(mesh.n_cells) if cell_ids is None else cell_ids
    k, j, i = np.unravel_index(cell_ids, np.maximum(dims - 1, 1)[::-1])
    steps = np.minimum(dims - 1, 1)
    corners = np.stack(
        [
            points[k + dk, j + dj, i + di]
            for dk in (0, steps[2])
            for dj in (0, steps[1])
            for di in (0, steps[0])
        ]
    )
    # Corner index bits are (i, j, k); an edge joins two corners differing in one bit,
    # and corners along a singleton axis coincide with the ones at bit zero
    singleton = sum(bit for axis, bit in enumerate((1, 2, 4)) if not steps[axis])
    edges = [
        (a, a | bit)
        for axis, bit in enumerate((1, 2, 4))
        if steps[axis]
        for a in range(8)
        if not a & (bit | singleton)
    ]
    lengths = [np.linalg.norm(corners[b] - corners[a], axis=1) for a, b in edges]
    return np.concatenate(lengths) if lengths else np.empty(0)


def _edges_per_axis(n_axes: int) -> int:
    """Return how many edges a hexahedron, quadrilateral, or line has along each axis."""
    return 2 ** (n_axes - 1) if n_axes else 0


def _polydata_edge_lengths(
    mesh: pv.PolyData, cell_ids: NumpyArray[int] | None
) -> NumpyArray[float]:
    """Return the edges of the vertex, line, polygon, and strip cells of polydata."""
    cell_arrays = [mesh.GetVerts(), mesh.GetLines(), mesh.GetPolys(), mesh.GetStrips()]
    base_types = [
        pv.CellType.POLY_VERTEX,
        pv.CellType.POLY_LINE,
        pv.CellType.POLYGON,
        pv.CellType.TRIANGLE_STRIP,
    ]
    offsets = [_vtk.vtk_to_numpy(ca.GetOffsetsArray()) for ca in cell_arrays]
    connectivity = np.concatenate(
        [_vtk.vtk_to_numpy(ca.GetConnectivityArray()) for ca in cell_arrays]
    )
    shifts = np.cumsum([0, *(o[-1] for o in offsets[:-1])])
    all_offsets = np.concatenate(
        [[0], *(o[1:] + shift for o, shift in zip(offsets, shifts, strict=True))]
    )
    cell_types = np.concatenate(
        [np.full(o.size - 1, t, dtype=np.uint8) for o, t in zip(offsets, base_types, strict=True)]
    )
    return _cell_array_edge_lengths(
        mesh,
        cell_types=cell_types,
        offsets=all_offsets,
        connectivity=connectivity,
        cell_ids=cell_ids,
    )


_EDGE_CHUNK_CELLS = 1_000_000


def _cell_array_edge_lengths(
    mesh: DataSet,
    *,
    cell_types: NumpyArray[np.uint8],
    offsets: NumpyArray[int],
    connectivity: NumpyArray[int],
    cell_ids: NumpyArray[int] | None,
) -> NumpyArray[float]:
    """Return the edges of cells described by cell types and a cell array."""
    cell_ids = np.arange(cell_types.size) if cell_ids is None else np.asarray(cell_ids)
    types = cell_types[cell_ids].astype(np.int64)
    sizes = np.diff(offsets)[cell_ids]
    starts = offsets[cell_ids]
    points = mesh.points
    lengths: list[NumpyArray[float]] = []
    # Cells of one type and size share a local edge table, so process them together
    base = int(sizes.max(initial=0)) + 1
    keys = types * base + sizes
    for key in np.unique(keys):
        selection = np.flatnonzero(keys == key)
        cell_type, size = divmod(int(key), base)
        if cell_type == pv.CellType.POLYHEDRON:
            lengths.extend(_cell_edges(mesh.GetCell(int(i))) for i in cell_ids[selection])
            continue
        edges = _local_edge_table(mesh, int(cell_ids[selection[0]]))
        if edges.size == 0:
            continue
        for chunk in np.array_split(selection, max(1, selection.size // _EDGE_CHUNK_CELLS)):
            cell_points = points[connectivity[starts[chunk][:, None] + np.arange(size)]]
            lengths.append(
                np.linalg.norm(
                    cell_points[:, edges[:, 0]] - cell_points[:, edges[:, 1]], axis=2
                ).ravel()
            )
    return np.concatenate(lengths).astype(float) if lengths else np.empty(0, dtype=float)


def _local_edge_table(mesh: DataSet, cell_id: int) -> NumpyArray[int]:
    """Return the endpoints of each edge of a cell as indices into the cell's points."""
    cell = _vtk.vtkGenericCell()
    mesh.GetCell(cell_id, cell)
    point_ids = cell.GetPointIds()
    n_points = point_ids.GetNumberOfIds()
    if cell.GetCellDimension() == 1:
        return _curve_edge_table(cell.GetCellType(), n_points)
    # Relabel the copied cell's point ids so its edges report local indices
    for i in range(n_points):
        point_ids.SetId(i, i)
    edges = []
    for edge_id in range(cell.GetNumberOfEdges()):
        edge_point_ids = cell.GetEdge(edge_id).GetPointIds()
        edges.append((edge_point_ids.GetId(0), edge_point_ids.GetId(1)))
    return np.array(edges, dtype=int).reshape(-1, 2)


def _curve_edge_table(cell_type: int, n_points: int) -> NumpyArray[int]:
    """Return the segments of a 1D cell, which has no edges of its own."""
    if cell_type == pv.CellType.POLY_LINE:
        return np.column_stack([np.arange(n_points - 1), np.arange(1, n_points)])
    return np.array([[0, 1]]) if n_points > 1 else np.empty((0, 2), dtype=int)


def _cell_edges(cell: _vtk.vtkCell) -> NumpyArray[float]:
    """Return the length of every edge of a single cell."""
    lengths: list[float] = []
    for edge_id in range(cell.GetNumberOfEdges()):
        edge_points = cell.GetEdge(edge_id).GetPoints()
        lengths.append(
            np.linalg.norm(np.subtract(edge_points.GetPoint(1), edge_points.GetPoint(0)))
        )
    return np.array(lengths, dtype=float)
