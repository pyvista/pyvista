"""Edge lengths of dataset cells and the percentile the voxelize filters use."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista_validation as _validation

import pyvista as pv
from pyvista import _vtk

if TYPE_CHECKING:
    import numpy.typing as npt

    from pyvista import DataSet

_EDGE_CHUNK_CELLS = 1_000_000


def _cell_length_percentile(mesh: DataSet, percentile: float, sample_size: int) -> float:
    """Return a percentile of the nonzero edge lengths of a seeded random sample of cells."""
    percentile = _validation.validate_number(
        percentile, must_be_in_range=[0.0, 1.0], name='cell_length_percentile'
    )
    sample_size = _validation.validate_number(
        sample_size,
        must_be_integer=True,
        must_be_in_range=[1, np.inf],
        dtype_out=int,
        name='cell_length_sample_size',
    )
    cell_ids = None
    if isinstance(mesh, pv.ImageData):
        # Every cell of an image is identical, so one cell measures them all
        cell_ids = np.zeros(1, dtype=int)
    elif sample_size < mesh.n_cells:
        cell_ids = np.sort(
            np.random.default_rng(0).choice(mesh.n_cells, sample_size, replace=False)
        )
    lengths = _cell_edge_lengths(mesh, cell_ids)
    lengths = lengths[lengths > 0]
    return float(np.quantile(lengths, percentile)) if lengths.size else 0.0


def _cell_edge_lengths(
    mesh: DataSet, cell_ids: npt.NDArray[np.signedinteger] | None = None
) -> npt.NDArray[np.floating]:
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


def _image_edge_lengths(
    mesh: pv.ImageData, cell_ids: npt.NDArray[np.signedinteger] | None
) -> npt.NDArray[np.float64]:
    """Return the four edges per axis shared by every cell of an image."""
    spacing = np.array(mesh.spacing, dtype=float)[np.array(mesh.dimensions) > 1]
    n_cells = mesh.n_cells if cell_ids is None else len(cell_ids)
    return np.repeat(spacing, _edges_per_axis(spacing.size) * n_cells)


def _rectilinear_edge_lengths(
    mesh: pv.RectilinearGrid, cell_ids: npt.NDArray[np.signedinteger] | None
) -> npt.NDArray[np.float64]:
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
    mesh: pv.StructuredGrid, cell_ids: npt.NDArray[np.signedinteger] | None
) -> npt.NDArray[np.floating]:
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
    mesh: pv.PolyData, cell_ids: npt.NDArray[np.signedinteger] | None
) -> npt.NDArray[np.float64]:
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


def _cell_array_edge_lengths(
    mesh: DataSet,
    *,
    cell_types: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.signedinteger],
    connectivity: npt.NDArray[np.signedinteger],
    cell_ids: npt.NDArray[np.signedinteger] | None,
) -> npt.NDArray[np.float64]:
    """Return the edges of cells described by cell types and a cell array."""
    cell_ids = np.arange(cell_types.size) if cell_ids is None else np.asarray(cell_ids)
    types = cell_types[cell_ids].astype(np.int64)
    sizes = np.diff(offsets)[cell_ids]
    starts = offsets[cell_ids]
    points = mesh.points
    lengths: list[npt.NDArray[np.floating]] = []
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


def _local_edge_table(mesh: DataSet, cell_id: int) -> npt.NDArray[np.signedinteger]:
    """Return the endpoints of each edge of a cell as indices into the cell's points."""
    cell = _vtk.vtkGenericCell()
    if isinstance(mesh, pv.ExplicitStructuredGrid):
        # Every cell is a hexahedron, and a hidden one would read as an empty cell
        cell.SetCellTypeToHexahedron()
    else:
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


def _curve_edge_table(cell_type: int, n_points: int) -> npt.NDArray[np.signedinteger]:
    """Return the segments of a 1D cell, which has no edges of its own."""
    if cell_type == pv.CellType.POLY_LINE:
        return np.column_stack([np.arange(n_points - 1), np.arange(1, n_points)])
    return np.array([[0, 1]]) if n_points > 1 else np.empty((0, 2), dtype=int)


def _cell_edges(cell: _vtk.vtkCell) -> npt.NDArray[np.float64]:
    """Return the length of every edge of a single cell."""
    lengths: list[float] = []
    for edge_id in range(cell.GetNumberOfEdges()):
        edge_points = cell.GetEdge(edge_id).GetPoints()
        lengths.append(
            np.linalg.norm(np.subtract(edge_points.GetPoint(1), edge_points.GetPoint(0)))
        )
    return np.array(lengths, dtype=float)
