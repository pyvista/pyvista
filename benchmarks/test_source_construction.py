"""Benchmarks for the geometric sources."""

from __future__ import annotations

import pyvista as pv


def test_solid_sphere(benchmark):
    """Build a solid sphere."""
    assert benchmark(pv.SolidSphere).n_cells


def test_sphere(benchmark):
    """Build a sphere surface."""
    assert benchmark(pv.Sphere).n_cells


def test_image_data(benchmark):
    """Build an image dataset."""
    assert benchmark(pv.ImageData, dimensions=(10, 10, 10)).n_points
