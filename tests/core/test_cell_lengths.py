from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.utilities import _cell_lengths
from pyvista.core.utilities._cell_lengths import _cell_edge_lengths
from pyvista.core.utilities._cell_lengths import _cell_length_percentile

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _edge_lengths_per_cell(mesh):
    """Compute the edge lengths of every cell one cell at a time."""
    per_cell = []
    for i in range(mesh.n_cells):
        cell = mesh.GetCell(i)
        points = cell.GetPoints()
        if cell.GetCellDimension() == 1:
            n_segments = points.GetNumberOfPoints() - 1
            if cell.GetCellType() != pv.CellType.POLY_LINE:
                n_segments = 1
            pairs = [(points.GetPoint(k), points.GetPoint(k + 1)) for k in range(n_segments)]
        else:
            pairs = []
            for e in range(cell.GetNumberOfEdges()):
                edge_points = cell.GetEdge(e).GetPoints()
                pairs.append((edge_points.GetPoint(0), edge_points.GetPoint(1)))
        per_cell.append(np.array([np.linalg.norm(np.subtract(b, a)) for a, b in pairs]))
    return per_cell


def _mixed_polydata():
    """Create polydata with vertex, line, polygon, and strip cells."""
    poly = pv.PolyData()
    poly.points = np.random.default_rng(0).random((20, 3))
    poly.verts = [1, 0, 2, 1, 2]
    poly.lines = [2, 3, 4, 3, 5, 6, 7]
    poly.faces = [3, 8, 9, 10, 4, 11, 12, 13, 14]
    poly.strips = [5, 15, 16, 17, 18, 19]
    return poly


def _every_cell_type():
    """Create an unstructured grid with one cell of every type."""
    grid = pv.UnstructuredGrid()
    for name, make_cell in inspect.getmembers(examples.cells, inspect.isfunction):
        if name[0].isupper():
            grid = grid + make_cell()
    return grid


@pytest.mark.parametrize(
    'make_mesh',
    [
        pv.Sphere,
        _mixed_polydata,
        _every_cell_type,
        examples.load_hexbeam,
        examples.load_explicit_structured,
        examples.load_structured,
        lambda: pv.StructuredGrid(*np.meshgrid([0.0, 1.0, 3.0], [0.0, 2.0], [0.0])),
        lambda: pv.StructuredGrid(*np.meshgrid([0.0, 1.0, 3.0], [0.0], [0.0])),
        lambda: pv.RectilinearGrid([0, 1, 3], [0, 2, 3, 7], [5, 6]),
        lambda: pv.RectilinearGrid([0, 1, 3], [0, 2], [5]),
        lambda: pv.RectilinearGrid([0], [2], [5]),
        lambda: pv.ImageData(dimensions=(3, 4, 5), spacing=(1, 2, 3)),
        lambda: pv.ImageData(dimensions=(3, 4, 1), spacing=(1, 2, 3)),
        lambda: pv.ImageData(dimensions=(5, 1, 1), spacing=(2, 1, 1)),
        lambda: pv.ImageData(dimensions=(1, 1, 1)),
        lambda: pv.PointSet([[0.0, 0.0, 0.0]]),
        pv.UnstructuredGrid,
    ],
    ids=[
        'sphere',
        'mixed_polydata',
        'every_cell_type',
        'hexbeam',
        'explicit_structured',
        'structured',
        'structured_2d',
        'structured_1d',
        'rectilinear',
        'rectilinear_2d',
        'rectilinear_0d',
        'image_3d',
        'image_2d',
        'image_1d',
        'image_0d',
        'pointset',
        'empty',
    ],
)
def test_cell_edge_lengths(make_mesh):
    mesh = make_mesh()
    expected = _edge_lengths_per_cell(mesh)

    lengths = _cell_edge_lengths(mesh)
    assert lengths.dtype == np.float64
    assert np.allclose(np.sort(lengths), np.sort(np.concatenate(expected or [[]])))

    cell_ids = np.array([mesh.n_cells - 1, 0])[: mesh.n_cells]
    subset = _cell_edge_lengths(mesh, cell_ids)
    assert np.allclose(
        np.sort(subset), np.sort(np.concatenate([expected[i] for i in cell_ids] or [[]]))
    )


def test_cell_edge_lengths_hidden_cells():
    grid = examples.load_explicit_structured()
    hidden = grid.hide_cells(range(10), inplace=False)
    assert np.array_equal(_cell_edge_lengths(hidden), _cell_edge_lengths(grid))
    for cell_ids in ([0, 50], [50, 0]):
        ids = np.array(cell_ids)
        assert np.array_equal(_cell_edge_lengths(hidden, ids), _cell_edge_lengths(grid, ids))

    structured = examples.load_structured()
    hidden = structured.hide_cells(range(10), inplace=False)
    assert np.array_equal(_cell_edge_lengths(hidden), _cell_edge_lengths(structured))


def test_cell_edge_lengths_cells_without_edges():
    grid = _every_cell_type()
    no_edges = {
        pv.CellType.EMPTY_CELL,
        pv.CellType.VERTEX,
        pv.CellType.POLY_VERTEX,
        pv.CellType.CONVEX_POINT_SET,
    }
    with_edges = np.flatnonzero(~np.isin(grid.celltypes, list(no_edges)))
    without_edges = np.flatnonzero(np.isin(grid.celltypes, list(no_edges)))
    assert with_edges.size
    assert without_edges.size
    assert _cell_edge_lengths(grid, without_edges).size == 0
    assert _cell_edge_lengths(grid, with_edges).size == _cell_edge_lengths(grid).size


def test_cell_edge_lengths_empty_cells():
    faces = [3, 0, 1, 2, 0, 3, 2, 3, 4, 0]
    poly = pv.PolyData(np.random.default_rng(0).random((5, 3)), faces=faces)
    assert poly.n_cells == 4
    lengths = _cell_edge_lengths(poly)
    assert lengths.size == 6
    assert np.allclose(np.sort(lengths), np.sort(np.concatenate(_edge_lengths_per_cell(poly))))


def test_cell_edge_lengths_image_direction_matrix():
    image = pv.ImageData(dimensions=(3, 3, 3), spacing=(1, 2, 3))
    image.direction_matrix = pv.Transform().rotate_z(45).matrix[:3, :3]
    assert np.array_equal(np.sort(_cell_edge_lengths(image)), np.repeat([1.0, 2.0, 3.0], 4 * 8))


def test_cell_length_percentile(ant):
    lengths = _cell_edge_lengths(ant)
    assert _cell_length_percentile(ant, 0.0, ant.n_cells) == lengths.min()
    assert _cell_length_percentile(ant, 1.0, ant.n_cells) == lengths.max()
    assert _cell_length_percentile(ant, 0.5, ant.n_cells) == np.quantile(lengths, 0.5)

    # A sample is a subset of the full distribution, drawn with a fixed seed
    sampled = _cell_length_percentile(ant, 0.0, 10)
    assert lengths.min() <= sampled <= lengths.max()
    assert sampled in lengths
    assert _cell_length_percentile(ant, 0.0, 10) == sampled

    # Cells without edges contribute nothing
    assert _cell_length_percentile(pv.PointSet(ant.points).cast_to_polydata(), 0.5, 10) == 0.0

    # Zero-length edges are ignored
    collapsed = pv.PolyData(ant.points, faces=[3, 0, 0, 0, *ant.faces])
    assert _cell_length_percentile(collapsed, 0.0, collapsed.n_cells) == lengths.min()


def test_cell_length_percentile_image_measures_one_cell(mocker: MockerFixture):
    image = pv.ImageData(dimensions=(100, 100, 100), spacing=(1, 2, 3))
    spy = mocker.spy(_cell_lengths, '_cell_edge_lengths')
    assert _cell_length_percentile(image, 0.5, image.n_cells) == 2.0
    assert np.array_equal(spy.call_args[0][1], [0])
