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
from pyvista.core.celltype import CellType
from pyvista.core.filters import _get_output
from pyvista.core.filters import _update_alg

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyvista import DataSet
    from pyvista import ExplicitStructuredGrid
    from pyvista import ImageData
    from pyvista import StructuredGrid
    from pyvista import UnstructuredGrid


def _mirroring_reorders() -> tuple[NDArray[np.intp], NDArray[np.bool_], NDArray[np.float64]]:
    """Return the source axes, reversals, and score weights of each mirroring axis reorder."""
    axes, flips = [], []
    for permutation in itertools.permutations(range(3)):
        for reversed_axes in itertools.product((False, True), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[permutation, range(3)] = np.where(reversed_axes, -1, 1)
            if np.linalg.det(matrix) < 0:
                axes.append(permutation)
                flips.append(reversed_axes)

    # A reorder sums how far each new axis points along the original axis in its place
    weights = np.zeros((9, len(axes)))
    for choice, (source, reverse) in enumerate(zip(axes, flips, strict=True)):
        weights[np.arange(3) * 3 + source, choice] = np.where(reverse, -1, 1)
    return np.array(axes), np.array(flips), weights


# New index axis i of a reorder is old axis _MIRROR_AXES[:, i], reversed if _MIRROR_FLIPS[:, i]
_MIRROR_AXES, _MIRROR_FLIPS, _MIRROR_WEIGHTS = _mirroring_reorders()

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


def _complete_frames(frames: NDArray[np.float64], real: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return unit index axes, filling the axes without extent to make a right-handed frame."""
    frames = frames.copy()
    columns = np.flatnonzero(real)
    with np.errstate(invalid='ignore', divide='ignore'):
        frames[..., columns] /= np.linalg.norm(frames[..., columns], axis=-2, keepdims=True)
        # A collapsed axis has no direction
        frames = np.nan_to_num(frames)
        if columns.size == 0:
            frames[:] = np.eye(3)
        elif columns.size == 1:
            # Take the next global axis, or the one after if the only axis lies along it
            axis = columns[0]
            first = frames[..., axis]
            candidates = np.eye(3)[[(axis + 1) % 3, (axis + 2) % 3]]
            # Round so that rounding error cannot break a tie between axes
            alignment = np.round(np.abs(first @ candidates.T), 8)
            other = candidates[np.argmin(alignment, axis=-1)]
            other -= np.sum(other * first, axis=-1, keepdims=True) * first
            frames[..., (axis + 1) % 3] = other / np.linalg.norm(other, axis=-1, keepdims=True)
            missing = (axis + 2) % 3
        elif columns.size == 2:
            missing = int(np.flatnonzero(~real)[0])
        if 0 < columns.size < 3:
            frames[..., missing] = np.cross(
                frames[..., (missing + 1) % 3], frames[..., (missing + 2) % 3]
            )
    return np.nan_to_num(frames)


def _map_frames(linear: NDArray[np.float64], frames: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return each frame's axes mapped by a linear transformation."""
    return np.tensordot(frames, linear, axes=([-2], [1])).swapaxes(-1, -2)


def _mirror_choice(
    frames: NDArray[np.float64], real: NDArray[np.bool_], matrix: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Return the mirroring reorder that keeps each frame of reflected index axes closest."""
    # Frames that all match, like the cells of a regular grid, share one choice
    if frames.ndim == 3 and len(frames) > 1:
        spread = np.abs(frames - frames[0]).max()
        if spread <= 1e-8 * np.abs(frames[0]).max():
            return np.full(len(frames), _mirror_choice(frames[0], real, matrix))

    linear = matrix[:3, :3]
    # Map the reflected axes back onto the original ones
    original = _complete_frames(_map_frames(np.linalg.inv(linear), frames), real)
    # Entry (i, j) is how far reflected axis j points along original axis i
    alignment = np.swapaxes(original, -1, -2) @ _map_frames(linear, original)
    scores = alignment.reshape(*alignment.shape[:-2], 9) @ _MIRROR_WEIGHTS
    # Scores within rounding error of the best tie, and a tie goes to the first reorder
    best = scores.max(axis=-1, keepdims=True)
    return np.argmax(scores >= best - 1e-8, axis=-1)


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
    dimensions = np.array(mesh.dimensions)
    choice = _mirror_choice(_structured_frame(mesh), dimensions > 1, matrix)
    axes, flips = _MIRROR_AXES[choice], _MIRROR_FLIPS[choice]
    extent = _reorder_extent(dimensions, mesh.offset, axes=axes, flips=flips)
    if isinstance(mesh, pv.ImageData):
        # Map each new index onto its old one, which keeps the origin where it is
        index_map = np.eye(4)
        index_map[:3, :3] = 0
        index_map[axes, range(3)] = np.where(flips, -1, 1)
        index_to_physical = mesh.index_to_physical_matrix @ index_map
        mesh.extent = extent
        mesh.index_to_physical_matrix = index_to_physical
    else:
        points = _reorder_block(mesh.points, dimensions, axes=axes, flips=flips)
        mesh.extent = extent
        mesh.points = points
    _reorder_structured_arrays(mesh, dimensions, axes=axes, flips=flips)


@functools.cache
def _block_edge_weights(local: tuple[tuple[int, ...], ...]) -> NDArray[np.float64]:
    """Return the weights that average a block cell's edges along each of its axes."""
    corners = np.array(local)
    weights = np.zeros((len(corners), 3))
    for axis in range(corners.shape[1]):
        upper = corners[:, axis] == 1
        weights[:, axis] = np.where(upper, 1, -1) / np.count_nonzero(upper)
    return weights


def _block_cell_frames(
    points: NDArray[np.float64], cells: NDArray[np.integer], local: tuple[tuple[int, ...], ...]
) -> NDArray[np.float64]:
    """Return the mean edge along each axis of block cells, given their point ids."""
    weights = _block_edge_weights(local)
    frames = np.empty((len(cells), 3, 3))
    # Gathering one coordinate at a time is faster than gathering whole points
    for coordinate in range(3):
        frames[:, coordinate] = points[:, coordinate][cells] @ weights
    return frames


@functools.cache
def _block_corner_orders(
    local: tuple[tuple[int, ...], ...], axes: tuple[int, ...]
) -> NDArray[np.intp]:
    """Return each mirroring reorder's old corner for each new corner of a cell along ``axes``."""
    corners = np.zeros((len(local), 3), dtype=int)
    corners[:, axes] = local
    lookup = {tuple(corner): i for i, corner in enumerate(corners)}
    orders = np.empty((len(_MIRROR_AXES), len(corners)), dtype=np.intp)
    for choice, (sources, flips) in enumerate(zip(_MIRROR_AXES, _MIRROR_FLIPS, strict=True)):
        # The cell's new axes are the new index axes that take one of its axes, in order
        new_axes = [axis for axis in range(3) if sources[axis] in axes]
        for i, corner in enumerate(local):
            old = np.zeros(3, dtype=int)
            for index, axis in zip(corner, new_axes, strict=True):
                old[sources[axis]] = 1 - index if flips[axis] else index
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
    inverse: NDArray[np.float64] = np.linalg.inv(matrix[:3, :3])
    points = mesh.points
    for cell_type, local in _BLOCK_CORNERS.items():
        ids = np.flatnonzero(celltypes == cell_type)
        if ids.size == 0:
            continue
        n_corners, n_axes = len(local), len(local[0])
        # Read each cell's points in the order the input lists them
        offsets, input_connectivity = original
        if ids.size == celltypes.size and input_connectivity.size == ids.size * n_corners:
            # Cells of a single type are laid out back to back
            rows = np.arange(input_connectivity.size).reshape(-1, n_corners)
            cells = input_connectivity.reshape(-1, n_corners)
        else:
            rows = offsets[ids, np.newaxis] + np.arange(n_corners)
            cells = input_connectivity[rows]
        frames = _block_cell_frames(points, cells, local)

        # A flat cell takes the index axes it would have in an axis-aligned grid
        if n_axes == 3:
            placements = np.zeros(len(ids), dtype=int)
        else:
            unreflected = _map_frames(inverse, frames)
            direction = (
                unreflected[..., 0]
                if n_axes == 1
                else np.cross(unreflected[..., 0], unreflected[..., 1])
            )
            placements = np.argmax(np.round(np.abs(direction), 8), axis=-1)

        # The reflected cells hold the same number of points in the same places
        connectivity = reflected[1]
        for placement in np.unique(placements):
            axes = {
                1: (int(placement),),
                2: tuple(axis for axis in range(3) if axis != placement),
                3: (0, 1, 2),
            }[n_axes]
            subset = placements == placement
            # Index every cell with a slice when they share a placement, which copies nothing
            selected = slice(None) if subset.all() else np.flatnonzero(subset)
            placed = np.zeros((np.count_nonzero(subset), 3, 3))
            placed[..., axes] = frames[selected][..., :n_axes]
            real = np.isin(range(3), axes)
            choices = _mirror_choice(placed, real, matrix)
            orders = _block_corner_orders(local, axes)
            unique_choices = np.unique(choices)
            for choice in unique_choices:
                chosen = (
                    selected
                    if unique_choices.size == 1
                    else np.flatnonzero(subset)[choices == choice]
                )
                connectivity[rows[chosen]] = cells[chosen][:, orders[choice]]


def _mirror_explicit_structured(mesh: ExplicitStructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Reorder a reflected explicit structured grid's cells in place to keep them oriented."""
    local = _BLOCK_CORNERS[CellType.HEXAHEDRON]
    hexahedra = _vtk.vtk_to_numpy(mesh.GetCells().GetConnectivityArray()).reshape(-1, 8)
    if hexahedra.size == 0:
        return
    # The mean cell frame weighs each point by how often each corner of a cell lands on it
    counts = np.stack([np.bincount(corner, minlength=mesh.n_points) for corner in hexahedra.T])
    frame = mesh.points.T @ (_block_edge_weights(local).T @ counts).T / len(hexahedra)
    choice = _mirror_choice(frame, np.ones(3, dtype=bool), matrix)
    axes, flips = _MIRROR_AXES[choice], _MIRROR_FLIPS[choice]

    # Reorder the cells like a block, then each cell's points like a block cell
    cell_ids = _reorder_block(
        np.arange(mesh.n_cells), np.array(mesh.dimensions) - 1, axes=axes, flips=flips
    )
    hexahedra = hexahedra[cell_ids[:, np.newaxis], _block_corner_orders(local, (0, 1, 2))[choice]]
    # The extent property keeps the dimensions, which the new cells change
    mesh.SetExtent(*_reorder_extent(mesh.dimensions, mesh.offset, axes=axes, flips=flips))
    mesh.SetCells(pv.CellArray.from_regular_cells(hexahedra))
    for name in mesh.cell_data.keys():
        mesh.cell_data[name] = mesh.cell_data[name][cell_ids]
    if 'ConnectivityFlags' in mesh.cell_data:
        mesh.compute_connectivity(inplace=True)


_AXIS_ALIGNED_TYPES = [CellType.VOXEL, CellType.PIXEL]


def _check_axis_aligned_cells(mesh: UnstructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Raise if a transformation would turn voxels or pixels away from the axes."""
    linear = matrix[:3, :3]
    if not np.isfinite(linear).all():
        return
    # A transformation that keeps cells on the axes maps each axis onto a single axis
    with np.errstate(invalid='ignore', divide='ignore'):
        columns = linear / np.linalg.norm(linear, axis=0)
    if np.all(np.count_nonzero(~np.isclose(columns, 0, atol=1e-8), axis=0) == 1):
        return
    if np.isin(mesh.celltypes, _AXIS_ALIGNED_TYPES).any():
        msg = (
            'Cannot transform VOXEL or PIXEL cells off the axes, since these cells must stay '
            'aligned with them.\nConvert them to HEXAHEDRON and QUAD cells first with '
            '`generalize_cells`.'
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


def _mirror_unstructured(mesh: UnstructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Reorder a reflected unstructured grid's cells in place to keep them oriented."""
    # Read the cells without caching arrays on the cells the mesh shares with the input
    input_cells = mesh.GetCells()
    original = (
        _vtk.vtk_to_numpy(input_cells.GetOffsetsArray()),
        _vtk.vtk_to_numpy(input_cells.GetConnectivityArray()),
    )
    celltypes = _vtk.vtk_to_numpy(mesh._get_cell_types_array())
    if np.isin(celltypes, list(_BLOCK_CORNERS)).all():
        # Copy the cells, which the mesh shares with the input, to reorder them like a grid
        reordered = _copy_cells(mesh)
    else:
        # Take the cells vtkReflectionFilter reorders, then reorder the block cells like a grid
        structure = pv.UnstructuredGrid()
        structure.copy_structure(mesh)
        alg = _vtk.vtkReflectionFilter()
        alg.SetInputData(structure)
        alg.SetCopyInput(False)
        _update_alg(alg)
        reordered = _get_output(alg, output_type=pv.UnstructuredGrid)
    _set_cells(mesh, reordered)
    cells = mesh.GetCells()
    reflected = (
        _vtk.vtk_to_numpy(cells.GetOffsetsArray()),
        _vtk.vtk_to_numpy(cells.GetConnectivityArray()),
    )
    _mirror_block_cells(mesh, celltypes, original=original, reflected=reflected, matrix=matrix)
    cells.GetConnectivityArray().Modified()


def _align_voxels_and_pixels(mesh: UnstructuredGrid, matrix: NDArray[np.float64]) -> None:
    """Reorder the corners of voxels and pixels that a rotation turned onto other axes."""
    # Cells whose axes stay in place are still valid, even with their axes reversed
    if np.all(np.diagonal(matrix[:3, :3]) != 0):
        return
    celltypes = _vtk.vtk_to_numpy(mesh._get_cell_types_array())
    if not np.isin(celltypes, _AXIS_ALIGNED_TYPES).any():
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
        rows = offsets[ids, np.newaxis] + np.arange(len(_BLOCK_CORNERS[cell_type]))
        point_ids = connectivity[rows]
        corners = points[point_ids]
        # A corner is at the upper end of each axis along which it lies past the cell's center
        upper = (corners > corners.mean(axis=1, keepdims=True)).astype(int)
        if cell_type == CellType.VOXEL:
            index = upper @ [1, 2, 4]
        else:
            # A pixel's corners vary along the two axes its plane spans, in order
            normal = np.argmin(np.ptp(corners, axis=1), axis=-1)
            spanned = np.array([[1, 2], [0, 2], [0, 1]])[normal]
            index = np.take_along_axis(upper, spanned[:, np.newaxis, :], axis=2) @ [1, 2]
        # Voxels and pixels list their corners along x first, then y, then z
        ordered = np.empty_like(point_ids)
        np.put_along_axis(ordered, index, point_ids, axis=1)
        connectivity[rows] = ordered
    cells.GetConnectivityArray().Modified()


def _mirror_cells(mesh: DataSet, matrix: NDArray[np.float64]) -> DataSet:
    """Return a reflected mesh with its cells reordered so they keep their orientation."""
    if isinstance(mesh, pv.UnstructuredGrid):
        _mirror_unstructured(mesh, matrix)
    elif isinstance(mesh, pv.PolyData):
        mesh = mesh._reverse_sense(
            reverse_cells=True, reverse_normals=False, inplace=False, progress_bar=False
        )
    elif isinstance(mesh, pv.ExplicitStructuredGrid):
        _mirror_explicit_structured(mesh, matrix)
    elif isinstance(mesh, pv.StructuredGrid):
        _mirror_structured(mesh, matrix)
    return mesh
