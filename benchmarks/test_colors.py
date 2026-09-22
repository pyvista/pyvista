"""Benchmarks for color parsing."""

from __future__ import annotations

import pyvista as pv


def test_color_from_name(benchmark):
    """Parse a named color."""
    assert benchmark(pv.Color, 'cornflowerblue').int_rgb


def test_color_from_hex(benchmark):
    """Parse a hex color."""
    assert benchmark(pv.Color, '#4682b4').int_rgb
