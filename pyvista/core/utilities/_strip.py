"""Hand :vtk:`vtkStripper` only the cells it handles soundly, and trace its output back."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pyvista as pv
from pyvista import _vtk

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyvista import PolyData


def _cells(connectivity: NDArray[np.int64], sizes: NDArray[np.int64]) -> pv.CellArray:
    """Return a cell array holding cells given by their point ids and sizes."""
    return pv.CellArray.from_arrays(np.concatenate([[0], np.cumsum(sizes)]), connectivity)


def _cell_windows(
    connectivity: NDArray[np.integer],
    offsets: NDArray[np.integer],
    width: int,
    *,
    first_id: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return every run of ``width`` consecutive point ids in each cell and the cell id of each.

    The ids of each run are sorted, so a run read backwards matches itself.
    """
    sizes = np.diff(offsets)
    counts = np.maximum(sizes - width + 1, 0)
    owners = np.repeat(np.arange(counts.size, dtype=np.int64) + first_id, counts)
    # number the runs within each cell, then shift them to where the cell starts
    local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    starts = np.repeat(offsets[:-1], counts) + local
    runs = connectivity[starts[:, np.newaxis] + np.arange(width)].astype(np.int64)
    return np.sort(runs, axis=1), owners


def _cell_rows(
    connectivity: NDArray[np.integer], offsets: NDArray[np.integer], width: int
) -> NDArray[np.int64]:
    """Return the sorted point ids of each cell, padded to ``width`` columns."""
    sizes = np.diff(offsets)
    rows = np.full((sizes.size, width), np.iinfo(np.int64).max, dtype=np.int64)
    owners = np.repeat(np.arange(sizes.size), sizes)
    rows[owners, np.arange(connectivity.size) - np.repeat(offsets[:-1], sizes)] = connectivity
    # the padding is the largest id, so it stays at the end of each sorted row
    return np.sort(rows, axis=1)


def _occurrences(keys: NDArray[np.intp]) -> NDArray[np.intp]:
    """Return how many earlier entries share each key."""
    order = np.argsort(keys, kind='stable')
    ordered = keys[order]
    starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))
    first = np.repeat(starts, np.diff(np.concatenate([starts, [keys.size]])))
    rank = np.empty_like(keys)
    rank[order] = np.arange(keys.size) - first
    return rank


def _match_runs(
    runs: NDArray[np.int64], owners: NDArray[np.int64], targets: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return the owner of the run equal to each target, or ``-1`` where none is."""
    if targets.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    _, inverse = np.unique(np.concatenate([runs, targets]), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    keys, wanted = inverse[: runs.shape[0]], inverse[runs.shape[0] :]
    # the k-th copy of a repeated run is matched to the k-th copy among the runs
    count = np.bincount(keys, minlength=inverse.max() + 1)
    rank = np.minimum(_occurrences(wanted), count[wanted] - 1)
    stride = int(count.max(initial=0)) + 1
    codes = keys * stride + _occurrences(keys)
    order = np.argsort(codes)
    wanted_codes = wanted * stride + rank
    position = np.minimum(np.searchsorted(codes[order], wanted_codes), order.size - 1)
    found = codes[order][position] == wanted_codes
    return np.where(found, owners[order][position], -1)


def _loose_cells(mesh: PolyData) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return the lines with no two distinct points and the triangles with a repeated point."""
    runs, owners = _cell_windows(mesh.line_connectivity, mesh.line_offsets, 2)
    lines = np.ones(mesh.n_lines, dtype=bool)
    lines[owners[runs[:, 0] != runs[:, 1]]] = False
    offsets = mesh.face_offsets
    polys = np.zeros(offsets.size - 1, dtype=bool)
    triangles = np.flatnonzero(np.diff(offsets) == 3)
    corners = mesh.face_connectivity[offsets[triangles, np.newaxis] + np.arange(3)]
    polys[triangles] = (
        (corners[:, 0] == corners[:, 1])
        | (corners[:, 1] == corners[:, 2])
        | (corners[:, 0] == corners[:, 2])
    )
    return lines, polys


def _kept_cells(
    connectivity: NDArray[np.integer], offsets: NDArray[np.integer], keep: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the point ids and sizes of the cells marked to keep."""
    sizes = np.diff(offsets)
    return connectivity[np.repeat(keep, sizes)].astype(np.int64), sizes[keep]


def _strip_source(
    mesh: PolyData, *, join: bool
) -> tuple[PolyData, NDArray[np.bool_], NDArray[np.bool_]]:
    """Return the mesh to hand the stripper, and the lines and polygons it leaves out.

    The stripper reads past a line of one repeated point, drops a line of one point and
    mishandles a triangle with a repeated point, so those cells are left out and added
    back afterwards. When joining, longer polylines gain invented edges, so the lines are
    split into segments.
    """
    loose_lines, loose_polys = _loose_cells(mesh)
    if not (join and mesh.n_lines) and not loose_lines.any() and not loose_polys.any():
        return mesh, loose_lines, loose_polys
    first_poly = mesh.n_verts + mesh.n_lines
    if join:
        runs, owners = _cell_windows(
            mesh.line_connectivity, mesh.line_offsets, 2, first_id=mesh.n_verts
        )
        segments = runs[:, 0] != runs[:, 1]
        lines = _cells(runs[segments].ravel(), np.full(segments.sum(), 2))
        line_cells = owners[segments]
    else:
        lines = _cells(*_kept_cells(mesh.line_connectivity, mesh.line_offsets, ~loose_lines))
        line_cells = mesh.n_verts + np.flatnonzero(~loose_lines)
    faces = _cells(*_kept_cells(mesh.face_connectivity, mesh.face_offsets, ~loose_polys))
    # a shallow copy keeps the points, point data, active names and field data as they are
    source = mesh.copy(deep=False)
    source.cell_data.clear()
    source.lines = lines
    source.faces = faces
    ghosts = _vtk.vtkDataSetAttributes.GhostArrayName()
    if ghosts in mesh.cell_data:
        # carry the ghost flag of each input cell onto the cells made from it
        cells = np.concatenate(
            [
                np.arange(mesh.n_verts),
                line_cells,
                first_poly + np.flatnonzero(~loose_polys),
                first_poly + loose_polys.size + np.arange(mesh.n_strips),
            ]
        )
        source.cell_data[ghosts] = np.asarray(mesh.cell_data[ghosts])[cells]
    return source, loose_lines, loose_polys


def _restore_cells(
    mesh: PolyData,
    output: PolyData,
    *,
    join: bool,
    lines: NDArray[np.bool_],
    polys: NDArray[np.bool_],
) -> None:
    """Add back to a strip the lines and polygons of the mesh that were left out of it."""
    ghosts = _vtk.vtkDataSetAttributes.GhostArrayName()
    if ghosts in mesh.cell_data:
        # the stripper skips ghost cells, so the cells left out of it skip them too
        hidden = np.asarray(mesh.cell_data[ghosts]) != 0
        first_poly = mesh.n_verts + mesh.n_lines
        lines = lines & ~hidden[mesh.n_verts : first_poly]
        polys = polys & ~hidden[first_poly : first_poly + polys.size]
    connectivity = output.line_connectivity.astype(np.int64)
    sizes = np.diff(output.line_offsets)
    if join:
        owners = np.repeat(np.arange(sizes.size), sizes)
        # joining repeats the point where two polylines meet
        keep = np.ones(connectivity.size, dtype=bool)
        keep[1:] = (connectivity[1:] != connectivity[:-1]) | (owners[1:] != owners[:-1])
        connectivity = connectivity[keep]
        sizes = np.bincount(owners[keep], minlength=sizes.size)
    loose, loose_sizes = _kept_cells(mesh.line_connectivity, mesh.line_offsets, lines)
    output.lines = _cells(
        np.concatenate([connectivity, loose]), np.concatenate([sizes, loose_sizes])
    )
    if polys.any():
        loose, loose_sizes = _kept_cells(mesh.face_connectivity, mesh.face_offsets, polys)
        output.faces = _cells(
            np.concatenate([output.face_connectivity, loose]),
            np.concatenate([np.diff(output.face_offsets), loose_sizes]),
        )


def _original_cell_ids(mesh: PolyData, output: PolyData) -> NDArray[np.int64]:
    """Return the input cell of each piece of a strip, in the order its field data holds them."""
    n_polys = mesh.face_offsets.size - 1
    first_poly = mesh.n_verts + mesh.n_lines
    first_strip = first_poly + n_polys
    line_runs, line_owners = _cell_windows(
        mesh.line_connectivity, mesh.line_offsets, 2, first_id=mesh.n_verts
    )
    lines = _match_runs(
        line_runs,
        line_owners,
        _cell_windows(output.line_connectivity, output.line_offsets, 2)[0],
    )
    width = int(
        max(np.diff(mesh.face_offsets).max(initial=0), np.diff(output.face_offsets).max(initial=0))
    )
    polygons = _match_runs(
        _cell_rows(mesh.face_connectivity, mesh.face_offsets, width),
        first_poly + np.arange(n_polys, dtype=np.int64),
        _cell_rows(output.face_connectivity, output.face_offsets, width),
    )
    triangle_runs, triangle_owners = _cell_windows(
        mesh.face_connectivity, mesh.face_offsets, 3, first_id=first_poly
    )
    # only triangles become strips, so a polygon's own runs of three are not candidates
    triangles = np.diff(mesh.face_offsets)[triangle_owners - first_poly] == 3
    strip_runs, strip_owners = _cell_windows(
        mesh.strip_connectivity, mesh.strip_offsets, 3, first_id=first_strip
    )
    strips = _match_runs(
        np.concatenate([triangle_runs[triangles], strip_runs]),
        np.concatenate([triangle_owners[triangles], strip_owners]),
        _cell_windows(output.strip_connectivity, output.strip_offsets, 3)[0],
    )
    verts = np.arange(output.n_verts, dtype=np.int64)
    return np.concatenate([verts, lines, polygons, strips])
