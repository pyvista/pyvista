"""Benchmarks for the cell wrapper and its geometry accessors."""

from __future__ import annotations

import operator


def test_get_cell(sphere, benchmark):
    """Build a cell wrapper from a dataset."""
    assert benchmark(sphere.get_cell, 0).n_points


def test_cell_point_ids(triangle_cell, benchmark):
    """Read the point ids of a cell."""
    assert benchmark(operator.attrgetter('point_ids'), triangle_cell)


def test_cell_n_edges(hex_cell, benchmark):
    """Read the edge count of a cell."""
    assert benchmark(operator.attrgetter('n_edges'), hex_cell)


def test_cell_bounds(hex_cell, benchmark):
    """Read the bounds of a cell."""
    assert benchmark(operator.attrgetter('bounds'), hex_cell)


def test_cell_edges(hex_cell, benchmark):
    """Build the edge wrappers of a cell."""
    assert benchmark(operator.attrgetter('edges'), hex_cell)


def test_cell_faces(hex_cell, benchmark):
    """Build the face wrappers of a cell."""
    assert benchmark(operator.attrgetter('faces'), hex_cell)


def test_cell_get_edge(hex_cell, benchmark):
    """Build one edge wrapper of a cell."""
    assert benchmark(hex_cell.get_edge, 0).n_points


def test_cell_get_face(hex_cell, benchmark):
    """Build one face wrapper of a cell."""
    assert benchmark(hex_cell.get_face, 0).n_points
