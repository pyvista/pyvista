"""Benchmarks for the clip and slice filters."""

from __future__ import annotations

import pyvista as pv


def test_clip_box_image(image, benchmark):
    """Clip an image by a box."""
    assert benchmark(image.clip_box, (2, 12, 2, 12, 2, 12), invert=False).n_cells


def test_clip_box_unstructured(hex_grid, benchmark):
    """Clip an unstructured grid of hexahedra by a box."""
    assert benchmark(hex_grid.clip_box, (1, 6, 1, 6, 1, 6), invert=False).n_cells


def test_clip_plane_polydata(sphere, benchmark):
    """Clip a surface by a plane."""
    assert benchmark(sphere.clip, 'x').n_cells


def test_clip_plane_unstructured(hex_grid, benchmark):
    """Clip an unstructured grid by a plane."""
    assert benchmark(hex_grid.clip, 'x').n_cells


def test_clip_plane_crinkle(hex_grid, benchmark):
    """Crinkle clip an unstructured grid by a plane."""
    assert benchmark(hex_grid.clip, 'x', crinkle=True).n_cells


def test_clip_surface_image(image, benchmark):
    """Clip an image by a closed surface."""
    surface = pv.Sphere(radius=6, center=image.center)
    assert benchmark(image.clip_surface, surface).n_cells


def test_slice_image(image, benchmark):
    """Slice an image by an axis-aligned plane."""
    assert benchmark(image.slice, 'z').n_cells


def test_slice_orthogonal_image(image, benchmark):
    """Slice an image by three orthogonal planes."""
    assert benchmark(image.slice_orthogonal).n_blocks == 3
