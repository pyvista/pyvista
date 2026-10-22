"""Hand :vtk:`vtkStripper` only the cells it handles soundly, and trace its output back."""

from __future__ import annotations

import functools
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


def _sort_rows(rows: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return the rows with the ids of each in ascending order."""
    # pairwise swaps are far cheaper than a sort along a two or three wide axis
    if rows.shape[1] == 2:
        a, b = rows.T
        return np.column_stack([np.minimum(a, b), np.maximum(a, b)])
    if rows.shape[1] == 3:
        a, b, c = rows.T
        low, high = np.minimum(a, b), np.maximum(a, b)
        mid, high = np.minimum(high, c), np.maximum(high, c)
        low, mid = np.minimum(low, mid), np.maximum(low, mid)
        return np.column_stack([low, mid, high])
    return np.sort(rows, axis=1)


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
    ids = np.asarray(connectivity, dtype=np.int64)
    sizes = np.diff(offsets)
    if sizes.size and sizes[0] >= width and (sizes == sizes[0]).all():
        # cells of one size are a table, so their runs are windows over its rows
        size = int(sizes[0])
        rows = ids.reshape(sizes.size, size)
        runs = np.lib.stride_tricks.sliding_window_view(rows, width, axis=1)
        owners = np.repeat(np.arange(sizes.size, dtype=np.int64) + first_id, size - width + 1)
        return _sort_rows(runs.reshape(-1, width)), owners
    counts = np.maximum(sizes - width + 1, 0)
    owners = np.repeat(np.arange(counts.size, dtype=np.int64) + first_id, counts)
    # number the runs within each cell, then shift them to where the cell starts
    local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    starts = np.repeat(offsets[:-1], counts) + local
    return _sort_rows(ids[starts[:, np.newaxis] + np.arange(width)]), owners


def _cell_rows(
    connectivity: NDArray[np.integer], offsets: NDArray[np.integer], width: int
) -> NDArray[np.int64]:
    """Return the sorted point ids of each cell, padded to ``width`` columns."""
    ids = np.asarray(connectivity, dtype=np.int64)
    sizes = np.diff(offsets)
    if sizes.size and (sizes == width).all():
        return _sort_rows(ids.reshape(sizes.size, width))
    rows = np.full((sizes.size, width), np.iinfo(np.int64).max, dtype=np.int64)
    owners = np.repeat(np.arange(sizes.size), sizes)
    rows[owners, np.arange(ids.size) - np.repeat(offsets[:-1], sizes)] = ids
    # the padding is the largest id, so it stays at the end of each sorted row
    return _sort_rows(rows)


def _occurrences(keys: NDArray[np.intp]) -> NDArray[np.intp]:
    """Return how many earlier entries share each key."""
    order = np.argsort(keys, kind='stable')
    ordered = keys[order]
    starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))
    first = np.repeat(starts, np.diff(np.concatenate([starts, [keys.size]])))
    rank = np.empty_like(keys)
    rank[order] = np.arange(keys.size) - first
    return rank


def _row_codes(rows: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return one integer per row, equal only for equal rows."""
    top = int(rows.max(initial=0)) + 1
    if top ** rows.shape[1] < 2**63:
        codes = np.zeros(rows.shape[0], dtype=np.int64)
        for column in rows.T:
            codes = codes * top + column
        return codes
    return np.unique(rows, axis=0, return_inverse=True)[1].ravel().astype(np.int64)


def _match_runs(
    runs: NDArray[np.int64], owners: NDArray[np.int64], targets: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return the owner of the run equal to each target, or ``-1`` where none is."""
    if targets.shape[0] == 0 or runs.shape[0] == 0:
        return np.full(targets.shape[0], -1, dtype=np.int64)
    codes = _row_codes(np.concatenate([runs, targets]))
    keys, wanted = codes[: runs.shape[0]], codes[runs.shape[0] :]
    order = np.argsort(keys)
    ordered = keys[order]
    first = np.searchsorted(ordered, wanted)
    if not (ordered[1:] == ordered[:-1]).any():
        position = np.minimum(first, order.size - 1)
        found = ordered[position] == wanted
        return np.where(found, owners[order][position], -1)
    # the k-th copy of a repeated target is matched to the k-th copy among the runs
    order = np.argsort(keys, kind='stable')
    ordered = keys[order]
    count = np.searchsorted(ordered, wanted, side='right') - first
    position = first + np.minimum(_occurrences(wanted), np.maximum(count - 1, 0))
    position = np.minimum(position, order.size - 1)
    return np.where(count > 0, owners[order][position], -1)


def _kept_cells(
    connectivity: NDArray[np.integer], offsets: NDArray[np.integer], keep: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the point ids and sizes of the cells marked to keep."""
    sizes = np.diff(offsets)
    return connectivity[np.repeat(keep, sizes)].astype(np.int64), sizes[keep]


class _StripTrace:
    """The cells handed to the stripper for a mesh, and the way from its output back.

    Lines without two distinct points and triangles with a repeated point stay out of the
    stripper, and joining is fed two-point segments, since the stripper mishandles them.

    Parameters
    ----------
    mesh : PolyData
        Mesh to strip.

    join : bool
        Whether the stripper joins contiguous lines.

    """

    def __init__(self, mesh: PolyData, *, join: bool) -> None:
        self.mesh = mesh
        self.join = join

    @functools.cached_property
    def segments(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return the sorted two-point runs of the lines and the cell id of each."""
        mesh = self.mesh
        return _cell_windows(mesh.line_connectivity, mesh.line_offsets, 2, first_id=mesh.n_verts)

    @functools.cached_property
    def degenerate_lines(self) -> NDArray[np.bool_]:
        """Mark the lines with no two distinct points."""
        connectivity, offsets = self.mesh.line_connectivity, self.mesh.line_offsets
        sizes = np.diff(offsets)
        degenerate = np.ones(sizes.size, dtype=bool)
        if connectivity.size < 2:
            return degenerate
        owners = np.repeat(np.arange(sizes.size), sizes)
        # a pair spanning two cells is an edge of neither
        distinct = (connectivity[1:] != connectivity[:-1]) & (owners[1:] == owners[:-1])
        degenerate[owners[:-1][distinct]] = False
        return degenerate

    @functools.cached_property
    def degenerate_polys(self) -> NDArray[np.bool_]:
        """Mark the triangles with a repeated point."""
        connectivity, offsets = self.mesh.face_connectivity, self.mesh.face_offsets
        sizes = np.diff(offsets)
        if sizes.size and (sizes == 3).all():
            triangles: slice | NDArray[np.intp] = slice(None)
            corners = connectivity.reshape(-1, 3)
        else:
            triangles = np.flatnonzero(sizes == 3)
            corners = connectivity[offsets[triangles, np.newaxis] + np.arange(3)]
        polys = np.zeros(sizes.size, dtype=bool)
        polys[triangles] = (
            (corners[:, 0] == corners[:, 1])
            | (corners[:, 1] == corners[:, 2])
            | (corners[:, 0] == corners[:, 2])
        )
        return polys

    @functools.cached_property
    def source(self) -> PolyData:
        """Return the mesh to hand the stripper, which is the mesh itself when it is sound."""
        mesh, join = self.mesh, self.join
        degenerate_lines, degenerate_polys = self.degenerate_lines, self.degenerate_polys
        if (
            not (join and mesh.n_lines)
            and not degenerate_lines.any()
            and not degenerate_polys.any()
        ):
            return mesh
        first_poly = mesh.n_verts + mesh.n_lines
        if join:
            runs, owners = self.segments
            segments = runs[:, 0] != runs[:, 1]
            lines = _cells(runs[segments].ravel(), np.full(segments.sum(), 2))
            line_cells = owners[segments]
        else:
            lines = _cells(
                *_kept_cells(mesh.line_connectivity, mesh.line_offsets, ~degenerate_lines)
            )
            line_cells = mesh.n_verts + np.flatnonzero(~degenerate_lines)
        faces = _cells(*_kept_cells(mesh.face_connectivity, mesh.face_offsets, ~degenerate_polys))
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
                    first_poly + np.flatnonzero(~degenerate_polys),
                    first_poly + degenerate_polys.size + np.arange(mesh.n_strips),
                ]
            )
            source.cell_data[ghosts] = np.asarray(mesh.cell_data[ghosts])[cells]
        return source

    def restore(self, output: PolyData) -> None:
        """Add back to a strip the cells of the mesh that were left out of its source.

        Parameters
        ----------
        output : PolyData
            Output of the stripper, changed in place.

        """
        if self.source is self.mesh:
            return
        mesh = self.mesh
        lines, polys = self.degenerate_lines, self.degenerate_polys
        ghosts = _vtk.vtkDataSetAttributes.GhostArrayName()
        if ghosts in mesh.cell_data:
            # the stripper skips ghost cells, so the cells left out of it skip them too
            hidden = np.asarray(mesh.cell_data[ghosts]) != 0
            first_poly = mesh.n_verts + mesh.n_lines
            lines = lines & ~hidden[mesh.n_verts : first_poly]
            polys = polys & ~hidden[first_poly : first_poly + polys.size]
        connectivity = output.line_connectivity.astype(np.int64)
        sizes = np.diff(output.line_offsets)
        if self.join:
            owners = np.repeat(np.arange(sizes.size), sizes)
            # joining repeats the point where two polylines meet
            keep = np.ones(connectivity.size, dtype=bool)
            keep[1:] = (connectivity[1:] != connectivity[:-1]) | (owners[1:] != owners[:-1])
            connectivity = connectivity[keep]
            sizes = np.bincount(owners[keep], minlength=sizes.size)
        degenerate, degenerate_sizes = _kept_cells(
            mesh.line_connectivity, mesh.line_offsets, lines
        )
        output.lines = _cells(
            np.concatenate([connectivity, degenerate]), np.concatenate([sizes, degenerate_sizes])
        )
        if polys.any():
            degenerate, degenerate_sizes = _kept_cells(
                mesh.face_connectivity, mesh.face_offsets, polys
            )
            output.faces = _cells(
                np.concatenate([output.face_connectivity, degenerate]),
                np.concatenate([np.diff(output.face_offsets), degenerate_sizes]),
            )

    def original_cell_ids(self, output: PolyData) -> NDArray[np.int64]:
        """Return the input cell of each piece of a strip, in field data order.

        Parameters
        ----------
        output : PolyData
            Output of the stripper.

        Returns
        -------
        numpy.ndarray
            Input cell id of each vert, line segment, polygon and strip triangle.

        """
        mesh = self.mesh
        n_polys = mesh.n_faces
        first_poly = mesh.n_verts + mesh.n_lines
        first_strip = first_poly + n_polys
        empty = np.zeros(0, dtype=np.int64)
        lines = polygons = strips = empty
        if output.n_lines:
            lines = _match_runs(
                *self.segments,
                _cell_windows(output.line_connectivity, output.line_offsets, 2)[0],
            )
        if output.n_faces:
            width = int(
                max(
                    np.diff(mesh.face_offsets).max(initial=0),
                    np.diff(output.face_offsets).max(initial=0),
                )
            )
            polygons = _match_runs(
                _cell_rows(mesh.face_connectivity, mesh.face_offsets, width),
                first_poly + np.arange(n_polys, dtype=np.int64),
                _cell_rows(output.face_connectivity, output.face_offsets, width),
            )
        if output.n_strips:
            # only triangles become strips, so a polygon's own runs of three are not candidates
            sizes = np.diff(mesh.face_offsets)
            triangles = sizes == 3
            corners = np.asarray(mesh.face_connectivity, dtype=np.int64)
            triangle_runs = _sort_rows(corners[np.repeat(triangles, sizes)].reshape(-1, 3))
            triangle_owners = first_poly + np.flatnonzero(triangles)
            strip_runs, strip_owners = _cell_windows(
                mesh.strip_connectivity, mesh.strip_offsets, 3, first_id=first_strip
            )
            strips = _match_runs(
                np.concatenate([triangle_runs, strip_runs]),
                np.concatenate([triangle_owners, strip_owners]),
                _cell_windows(output.strip_connectivity, output.strip_offsets, 3)[0],
            )
        verts = np.arange(output.n_verts, dtype=np.int64)
        return np.concatenate([verts, lines, polygons, strips])
