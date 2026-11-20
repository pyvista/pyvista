"""Reorder reflected meshes so their cells keep their orientation."""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.core._structured_data import _reorder_block
from pyvista.core._structured_data import _reorder_extent
from pyvista.core._structured_data import _reorder_structured_arrays
from pyvista.core._vtk_utilities import _SUPPORTS_POLYHEDRON_FACE_CELL_ARRAYS
from pyvista.core.celltype import CellType
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from pyvista import DataSet
    from pyvista import ExplicitStructuredGrid
    from pyvista import ImageData
    from pyvista import StructuredGrid
    from pyvista import UnstructuredGrid


@functools.cache
def _mirroring_reorders(
    n_axes: int,
) -> tuple[NDArray[np.intp], NDArray[np.bool_], NDArray[np.float64]]:
    """Return the source axes, reversals, and score weights of each reorder of ``n_axes`` axes.

    New axis ``i`` of a reorder is old axis ``axes[i]``, reversed if ``flips[i]``. Each
    reorder undoes itself, so a second reflection restores each cell. A reorder of three
    axes mirrors them, while a cell of fewer axes leaves that to the axes it lacks.
    """
    axes, flips = [], []
    for permutation in itertools.permutations(range(n_axes)):
        for reversed_axes in itertools.product((False, True), repeat=n_axes):
            matrix = np.zeros((n_axes, n_axes))
            matrix[permutation, range(n_axes)] = np.where(reversed_axes, -1, 1)
            if np.array_equal(matrix, matrix.T) and (n_axes < 3 or np.linalg.det(matrix) < 0):
                axes.append(permutation)
                flips.append(reversed_axes)

    # A reorder sums how far each new axis points along the original axis in its place
    weights = np.zeros((n_axes * n_axes, len(axes)))
    for choice, (source, reverse) in enumerate(zip(axes, flips, strict=True)):
        weights[np.arange(n_axes) * n_axes + source, choice] = np.where(reverse, -1, 1)
    return (
        np.array(axes, dtype=np.intp).reshape(-1, n_axes),
        np.array(flips, dtype=bool).reshape(-1, n_axes),
        weights,
    )


# The local index of each corner of the linear cells whose points form a structured block
_BLOCK_CORNERS = {
    CellType.LINE: ((0,), (1,)),
    CellType.PIXEL: ((0, 0), (1, 0), (0, 1), (1, 1)),
    CellType.QUAD: ((0, 0), (1, 0), (1, 1), (0, 1)),
    CellType.VOXEL: tuple((i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)),
    CellType.HEXAHEDRON: (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ),
}  # fmt: skip


def _map_frames(linear: NDArray[np.float64], frames: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return each frame's axes mapped by a linear transformation, laid out like the input."""
    return (linear @ frames.reshape(3, -1)).reshape(frames.shape)


def _mirror_choice(
    frames: NDArray[np.float64],
    matrix: NDArray[np.float64],
    *,
    atol: float = 0.0,
    allowed: NDArray[np.bool_] | None = None,
) -> NDArray[np.intp]:
    """Return the reorder that keeps each frame of reflected axes closest to the original.

    Frames are laid out as ``(coordinate, axis, frame)`` or as one ``(coordinate, axis)``
    frame, with the axes a cell has. Frames that differ by at most ``atol`` count as matching,
    and ``allowed`` limits the reorders to choose from.
    """
    if frames.ndim == 2:
        return _mirror_choice(frames[..., np.newaxis], matrix, allowed=allowed)[0]
    # Frames that all match, like the cells of a regular grid, share one choice
    if frames.shape[-1] > 1:
        spread = np.abs(frames - frames[..., :1]).max()
        if spread <= max(1e-8 * np.abs(frames[..., 0]).max(), atol):
            choice = _mirror_choice(frames[..., 0], matrix, allowed=allowed)
            return np.full(frames.shape[-1], choice)

    n_axes = frames.shape[1]
    linear = matrix[:3, :3]
    # Map the reflected axes back onto the original ones, as unit vectors
    original = _map_frames(np.linalg.inv(linear), frames)
    with np.errstate(invalid='ignore', divide='ignore'):
        # A collapsed axis has no direction
        original = np.nan_to_num(original / np.sqrt((original**2).sum(axis=0)), copy=False)
    # Entry (i, j) is how far reflected axis j points along original axis i
    mapped = _map_frames(linear, original)
    alignment = np.sum(original[:, :, np.newaxis] * mapped[:, np.newaxis], axis=0)
    sources, flips, weights = _mirroring_reorders(n_axes)
    scores = weights.T @ alignment.reshape(n_axes * n_axes, -1)
    if allowed is not None:
        scores[~allowed] = -np.inf
    # Scores within rounding error of the best tie
    tied = scores >= scores.max(axis=0) - 1e-8
    if np.count_nonzero(tied, axis=0).max() > 1:
        # A tie goes to the reorder whose new axes point furthest along the global axes
        new_axes = mapped[:, sources] * np.where(flips, -1, 1)[np.newaxis, :, :, np.newaxis]
        scores = np.where(tied, new_axes.max(axis=0).sum(axis=1), -np.inf)
        tied = scores >= scores.max(axis=0) - 1e-8
    # A remaining tie goes to the first reorder
    return np.argmax(tied, axis=0)


def _structured_frame(mesh: ImageData | StructuredGrid) -> NDArray[np.float64]:
    """Return the mean step along each index axis of a structured mesh."""
    if isinstance(mesh, pv.ImageData):
        return mesh.index_to_physical_matrix[:3, :3]
    dimensions = np.array(mesh.dimensions)
    points = mesh.points.reshape(*dimensions[::-1], 3)
    frame = np.zeros((3, 3))
    for axis in np.flatnonzero(dimensions > 1):
        # The steps along an axis add up to the distance from its first to its last points
        span = np.take(points, -1, axis=2 - axis) - np.take(points, 0, axis=2 - axis)
        frame[:, axis] = span.reshape(-1, 3).mean(axis=0) / (dimensions[axis] - 1)
    return frame


def _mirror_structured(mesh: ImageData | StructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Reorder a reflected image or structured grid in place to keep its cells' orientation."""
    if mesh.n_points == 0:
        return
    dimensions = np.array(mesh.dimensions)
    sources, flips, _ = _mirroring_reorders(3)
    # An axis without extent has no direction and stays in place, as the grid's cells keep it
    real = dimensions > 1
    frame = _structured_frame(mesh)
    frame[:, ~real] = 0
    allowed = np.all(real[sources] == real, axis=1)
    choice = _mirror_choice(frame, matrix, allowed=allowed)
    axes, flips = sources[choice], flips[choice]
    extent = _reorder_extent(dimensions, mesh.offset, axes=axes, flips=flips)
    if isinstance(mesh, pv.ImageData):
        # Each new axis takes its old axis's spacing and direction, which keeps the origin
        direction = mesh.direction_matrix[:, axes] * np.where(flips, -1, 1)
        spacing = np.asarray(mesh.spacing)[axes]
        mesh.extent = extent
        mesh.spacing = spacing
        mesh.direction_matrix = direction
    else:
        points = _reorder_block(mesh.points, dimensions, axes=axes, flips=flips)
        mesh.extent = extent
        mesh.points = points
    _reorder_structured_arrays(mesh, dimensions, axes=axes, flips=flips)


@functools.cache
def _block_edge_weights(local: tuple[tuple[int, ...], ...]) -> NDArray[np.float64]:
    """Return the weights that average a block cell's edges along each of its axes."""
    corners = np.array(local)
    weights = np.zeros(corners.shape)
    for axis in range(corners.shape[1]):
        upper = corners[:, axis] == 1
        weights[:, axis] = np.where(upper, 1, -1) / np.count_nonzero(upper)
    return weights


def _block_cell_frames(
    points: NDArray[np.float64], cells: NDArray[np.integer], local: tuple[tuple[int, ...], ...]
) -> NDArray[np.float64]:
    """Return the mean edge along each axis of block cells as ``(coordinate, axis, cell)``."""
    weights = _block_edge_weights(local)
    frames = np.empty((3, weights.shape[1], len(cells)))
    # Gathering one coordinate at a time is faster than gathering whole points
    for coordinate in range(3):
        frames[coordinate] = (points[:, coordinate][cells] @ weights).T
    return frames


@functools.cache
def _block_corner_orders(local: tuple[tuple[int, ...], ...]) -> NDArray[np.intp]:
    """Return each reorder's old corner for each new corner of a block cell."""
    corners = np.array(local)
    lookup = {tuple(corner): i for i, corner in enumerate(corners)}
    sources, flips, _ = _mirroring_reorders(corners.shape[1])
    orders = np.empty((len(sources), len(corners)), dtype=np.intp)
    for choice, (source, reverse) in enumerate(zip(sources, flips, strict=True)):
        for i, corner in enumerate(corners):
            old = np.empty_like(corner)
            old[source] = np.where(reverse, 1 - corner, corner)
            orders[choice, i] = lookup[tuple(old)]
    return orders


def _mirror_block_cells(
    mesh: UnstructuredGrid,
    celltypes: NDArray[np.uint8],
    *,
    original: tuple[NDArray[np.integer], NDArray[np.integer]],
    reflected: tuple[NDArray[np.integer], NDArray[np.integer]],
    matrix: NDArray[np.float64],
) -> None:
    """Reorder the points of reflected cells that form a block, like a reflected grid's cells.

    The cells are read from the ``original`` offsets and connectivity, and written to the
    ``reflected`` ones.
    """
    points = mesh.points
    # Frames of a regular grid differ by the rounding of its points
    precision = np.finfo(np.result_type(points.dtype, np.float32)).eps
    atol = 4 * precision * float(np.abs(points).max(initial=0))
    offsets, input_connectivity = original
    # The reflected cells hold the same number of points in the same places
    connectivity = reflected[1]
    for cell_type, local in _BLOCK_CORNERS.items():
        ids = np.flatnonzero(celltypes == cell_type)
        if ids.size == 0:
            continue
        n_corners = len(local)
        # Read each cell's points in the order the input lists them, at its reflected rows
        uniform = ids.size == celltypes.size and input_connectivity.size == ids.size * n_corners
        if uniform:
            # Cells of a single type are laid out back to back
            cells = input_connectivity.reshape(-1, n_corners)
            target = connectivity.reshape(-1, n_corners)
        else:
            rows = reflected[0][ids, np.newaxis] + np.arange(n_corners)
            cells = input_connectivity[offsets[ids, np.newaxis] + np.arange(n_corners)]
        choices = _mirror_choice(_block_cell_frames(points, cells, local), matrix, atol=atol)
        orders = _block_corner_orders(local)
        unique_choices = [choices[0]] if np.all(choices == choices[0]) else np.unique(choices)
        for choice in unique_choices:
            # Index every cell with a slice when they share a choice, which copies nothing
            chosen = slice(None) if len(unique_choices) == 1 else np.flatnonzero(choices == choice)
            reordered = cells[chosen][:, orders[choice]]
            if uniform:
                target[chosen] = reordered
            else:
                connectivity[rows[chosen]] = reordered


def _mirror_explicit_structured(mesh: ExplicitStructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Reorder a reflected explicit structured grid's cells in place to keep them oriented."""
    local = _BLOCK_CORNERS[CellType.HEXAHEDRON]
    hexahedra = _vtk.vtk_to_numpy(mesh.GetCells().GetConnectivityArray()).reshape(-1, 8)
    if hexahedra.size == 0:
        return
    # The mean cell frame weighs each point by how often each corner of a cell lands on it
    counts = np.stack([np.bincount(corner, minlength=mesh.n_points) for corner in hexahedra.T])
    frame = mesh.points.T @ (_block_edge_weights(local).T @ counts).T / len(hexahedra)
    choice = _mirror_choice(frame, matrix)
    sources, reversals, _ = _mirroring_reorders(3)
    axes, flips = sources[choice], reversals[choice]

    # Reorder the cells like a block, then each cell's points like a block cell
    cell_ids = _reorder_block(
        np.arange(mesh.n_cells), np.array(mesh.dimensions) - 1, axes=axes, flips=flips
    )
    hexahedra = hexahedra[cell_ids[:, np.newaxis], _block_corner_orders(local)[choice]]
    # The extent property keeps the dimensions, which the new cells change
    mesh.SetExtent(*_reorder_extent(mesh.dimensions, mesh.offset, axes=axes, flips=flips))
    mesh.SetCells(pv.CellArray.from_regular_cells(hexahedra))
    for name in mesh.cell_data.keys():
        mesh.cell_data[name] = mesh.cell_data[name][cell_ids]
    if 'ConnectivityFlags' in mesh.cell_data:
        mesh.compute_connectivity(inplace=True)


_AXIS_ALIGNED_TYPES = [CellType.VOXEL, CellType.PIXEL]


def _has_types(celltypes: NDArray[np.uint8], types: Iterable[int]) -> NDArray[np.bool_]:
    """Return which cells have one of the types."""
    return np.logical_or.reduce([celltypes == cell_type for cell_type in types])


def _axis_images(matrix: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return which axes the transformation maps each axis onto, ignoring rounding error."""
    linear = matrix[:3, :3]
    with np.errstate(invalid='ignore'):
        # An axis scaled to zero maps onto no axis
        return np.abs(linear) > 1e-8 * np.sqrt((linear**2).sum(axis=0))


def _check_axis_aligned_cells(mesh: UnstructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Raise if a transformation would turn voxels or pixels away from the axes."""
    if not np.isfinite(matrix[:3, :3]).all():
        return
    # A transformation that keeps cells on the axes maps each axis onto at most one axis
    if np.all(np.count_nonzero(_axis_images(matrix), axis=0) <= 1):
        return
    if _has_types(mesh.celltypes, _AXIS_ALIGNED_TYPES).any():
        msg = (
            'Cannot transform VOXEL or PIXEL cells off the axes, since these cells must stay '
            'aligned with them.\nConvert them to HEXAHEDRON and QUAD cells first with '
            '`convert_axis_aligned_cells`.'
        )
        raise ValueError(msg)


def _copy_cells(mesh: UnstructuredGrid) -> UnstructuredGrid:
    """Return a grid with a deep copy of the mesh's cells and no points."""
    structure = pv.UnstructuredGrid()
    structure.copy_structure(mesh)
    structure.SetPoints(_vtk.vtkPoints())
    copied = pv.UnstructuredGrid()
    copied.DeepCopy(structure)
    return copied


def _set_cells(mesh: UnstructuredGrid, source: UnstructuredGrid) -> None:
    """Give the mesh the cells of another grid, keeping its own points."""
    points = mesh.GetPoints()
    mesh.copy_structure(source)
    mesh.SetPoints(points)


def _replace_cells(
    mesh: UnstructuredGrid, cells: pv.CellArray, celltypes: NDArray[np.uint8] | None = None
) -> None:
    """Give the mesh new cells, of the same types by default, keeping its polyhedron faces."""
    types = _vtk.numpy_to_vtk(mesh.celltypes if celltypes is None else celltypes, deep=True)
    if _SUPPORTS_POLYHEDRON_FACE_CELL_ARRAYS:
        faces = (mesh.GetPolyhedronFaceLocations(), mesh.GetPolyhedronFaces())
        mesh.SetPolyhedralCells(types, cells, *faces)
    else:
        locations = _vtk.numpy_to_vtkIdTypeArray(
            cells.cell_offsets[:-1].astype(pv.ID_TYPE), deep=True
        )
        mesh.SetCells(types, locations, cells, mesh.GetFaceLocations(), mesh.GetFaces())


# Cells that vtkReflectionFilter reverses into invalid cells, besides triangle strips
_REVERSED_TYPES = [CellType.BIQUADRATIC_TRIANGLE, CellType.QUADRATIC_POLYGON]


def _reversed_order(cell_type: int, size: int) -> NDArray[np.intp]:
    """Return the order of a cell's points that reverses it."""
    if cell_type == CellType.BIQUADRATIC_TRIANGLE:
        return np.array([0, 2, 1, 5, 4, 3, 6])
    # Reverse the corners from the first, then the edge midpoints between them
    n_corners = size // 2
    return np.concatenate((-np.arange(n_corners) % n_corners, size - 1 - np.arange(n_corners)))


def _offsets_from_sizes(sizes: NDArray[np.integer]) -> NDArray[np.int64]:
    """Return the offsets of cells of the given sizes."""
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return offsets


def _reverse_strips(
    sizes: NDArray[np.integer], starts: NDArray[np.integer], connectivity: NDArray[np.integer]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return each reversed triangle strip's size and the connectivity indices of its points.

    Reversing a strip of an even number of points keeps its orientation, so its first point
    is repeated to shift its triangles, and a repeated first point is dropped instead.
    """
    even = sizes % 2 == 0
    repeated = np.zeros(len(sizes), dtype=bool)
    long = sizes > 1
    repeated[long] = connectivity[starts[long]] == connectivity[starts[long] + 1]
    new_sizes = sizes + even - (~even & repeated)
    offsets = _offsets_from_sizes(new_sizes)
    positions = np.arange(offsets[-1]) - np.repeat(offsets[:-1], new_sizes)
    # Count down from the last point, holding on it once more for an even strip
    held = np.repeat(even, new_sizes) & (positions > 0)
    return new_sizes, np.repeat(starts + sizes - 1, new_sizes) - positions + held


def _reverse_cells(
    mesh: UnstructuredGrid,
    celltypes: NDArray[np.uint8],
    original: tuple[NDArray[np.integer], NDArray[np.integer]],
) -> None:
    """Reverse the reflected cells that vtkReflectionFilter reorders wrongly."""
    reversed_cells = _has_types(celltypes, [CellType.TRIANGLE_STRIP, *_REVERSED_TYPES])
    if not reversed_cells.any():
        return
    cells = mesh.GetCells()
    offsets = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    input_offsets, input_connectivity = original
    input_sizes = np.diff(input_offsets)
    sizes = np.where(reversed_cells, input_sizes, np.diff(offsets))
    strips = np.flatnonzero(celltypes == CellType.TRIANGLE_STRIP)
    strip_sizes, strip_source = _reverse_strips(
        input_sizes[strips], input_offsets[strips], input_connectivity
    )
    sizes[strips] = strip_sizes
    new_offsets = _offsets_from_sizes(sizes)
    # Index the reflected connectivity followed by the input's
    source = np.repeat(offsets[:-1] - new_offsets[:-1], sizes) + np.arange(new_offsets[-1])
    strip_offsets = _offsets_from_sizes(strip_sizes)
    rows = np.repeat(new_offsets[strips] - strip_offsets[:-1], strip_sizes) + np.arange(
        strip_offsets[-1]
    )
    source[rows] = connectivity.size + strip_source
    for cell_type in _REVERSED_TYPES:
        ids = np.flatnonzero(celltypes == cell_type)
        for size in np.unique(input_sizes[ids]):
            group = ids[input_sizes[ids] == size]
            rows = new_offsets[group, np.newaxis] + np.arange(size)
            order = _reversed_order(cell_type, int(size))
            source[rows] = connectivity.size + input_offsets[group, np.newaxis] + order
    merged = np.concatenate((connectivity, input_connectivity))
    _replace_cells(mesh, pv.CellArray.from_arrays(new_offsets, merged[source]))


def _mirror_unstructured(mesh: UnstructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Reorder a reflected unstructured grid's cells in place to keep them oriented."""
    # Read the cells without caching arrays on the cells the mesh shares with the input
    input_cells = mesh.GetCells()
    original = (
        _vtk.vtk_to_numpy(input_cells.GetOffsetsArray()),
        _vtk.vtk_to_numpy(input_cells.GetConnectivityArray()),
    )
    celltypes = _vtk.vtk_to_numpy(mesh._get_cell_types_array())
    if _has_types(celltypes, _BLOCK_CORNERS).all():
        # Copy the cells, which the mesh shares with the input, to reorder them like a grid
        reordered = _copy_cells(mesh)
    else:
        # Take the cells vtkReflectionFilter reorders, then reorder the block cells like a grid
        structure = pv.UnstructuredGrid()
        structure.copy_structure(mesh)
        alg = _vtk.vtkReflectionFilter()
        alg.SetInputData(structure)
        alg.SetCopyInput(False)
        # It warns about the cells it cannot reverse, which _reverse_cells reverses
        with pv.vtk_verbosity('off'), pv.VtkErrorCatcher(send_to_logging=False):
            _update_alg(alg)
        reordered = _get_output(alg, output_type=pv.UnstructuredGrid)
        _reverse_cells(reordered, celltypes, original)
    _set_cells(mesh, reordered)
    cells = mesh.GetCells()
    reflected = (
        _vtk.vtk_to_numpy(cells.GetOffsetsArray()),
        _vtk.vtk_to_numpy(cells.GetConnectivityArray()),
    )
    _mirror_block_cells(mesh, celltypes, original=original, reflected=reflected, matrix=matrix)
    cells.GetConnectivityArray().Modified()


def _align_voxels_and_pixels(mesh: UnstructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Reorder the corners of voxels and pixels that a transformation turned or reversed."""
    if np.all(np.diagonal(_axis_images(matrix))) and np.all(np.diagonal(matrix[:3, :3]) > 0):
        return
    celltypes = _vtk.vtk_to_numpy(mesh._get_cell_types_array())
    if not _has_types(celltypes, _AXIS_ALIGNED_TYPES).any():
        return

    # Copy the cells, which the mesh shares with the input, to reorder them
    _set_cells(mesh, _copy_cells(mesh))
    cells = mesh.GetCells()
    offsets = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    points = mesh.points
    for cell_type in _AXIS_ALIGNED_TYPES:
        ids = np.flatnonzero(celltypes == cell_type)
        if ids.size == 0:
            continue
        n_corners = len(_BLOCK_CORNERS[cell_type])
        uniform = ids.size == celltypes.size and connectivity.size == ids.size * n_corners
        if uniform:
            # Cells of a single type are laid out back to back
            target = connectivity.reshape(-1, n_corners)
            point_ids = target.copy()
        else:
            rows = offsets[ids, np.newaxis] + np.arange(n_corners)
            point_ids = connectivity[rows]
        if cell_type == CellType.VOXEL:
            spanned = None
        else:
            # A pixel's corners vary along the two axes its plane spans, in order
            extents = [np.ptp(points[:, axis][point_ids], axis=1) for axis in range(3)]
            spanned = np.array([[1, 2], [0, 2], [0, 1]])[np.argmin(extents, axis=0)]
        # Voxels and pixels list their corners along x first, then y, then z
        index = np.zeros(point_ids.shape, dtype=np.intp)
        for place in range(3 if spanned is None else 2):
            # Gathering one coordinate at a time is faster than gathering whole points
            coordinates = (
                points[:, place][point_ids]
                if spanned is None
                else points[point_ids, spanned[:, place, np.newaxis]]
            )
            # A corner is at the upper end of each axis along which it lies past the center
            index += (coordinates > coordinates.mean(axis=1, keepdims=True)) << place
        # A collapsed cell has no corner order to restore
        distinct = np.bitwise_or.reduce(1 << index, axis=1) == (1 << n_corners) - 1
        kept = slice(None) if distinct.all() else np.flatnonzero(distinct)
        if uniform:
            target[np.arange(len(target))[kept, np.newaxis], index[kept]] = point_ids[kept]
        else:
            ordered = np.empty_like(point_ids)
            ordered[np.arange(len(ids))[:, np.newaxis], index] = point_ids
            connectivity[rows[kept]] = ordered[kept]
    cells.GetConnectivityArray().Modified()


def _mirror_cells(mesh: DataSet, matrix: NDArray[np.float64]) -> DataSet:
    """Return a reflected mesh with its cells reordered so they keep their orientation."""
    if isinstance(mesh, pv.UnstructuredGrid):
        _mirror_unstructured(mesh, matrix)
    elif isinstance(mesh, pv.PolyData):
        strips = mesh.GetStrips()
        mesh = mesh._reverse_sense(
            reverse_cells=True, reverse_normals=False, inplace=False, progress_bar=False
        )
        if mesh.n_strips:
            # vtkReverseSense keeps the orientation of strips of an even number of points
            offsets = _vtk.vtk_to_numpy(strips.GetOffsetsArray())
            connectivity = _vtk.vtk_to_numpy(strips.GetConnectivityArray())
            sizes, source = _reverse_strips(np.diff(offsets), offsets[:-1], connectivity)
            mesh.strips = pv.CellArray.from_arrays(
                _offsets_from_sizes(sizes), connectivity[source]
            )
    elif isinstance(mesh, pv.ExplicitStructuredGrid):
        _mirror_explicit_structured(mesh, matrix)
    elif isinstance(mesh, pv.StructuredGrid):
        _mirror_structured(mesh, matrix)
    return mesh
