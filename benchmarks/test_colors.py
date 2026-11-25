"""Benchmarks for color parsing."""

from __future__ import annotations

import pyvista as pv
from pyvista.plotting.colors import get_cmap_safe


def test_color_from_name(benchmark):
    """Parse a named color."""
    assert benchmark(pv.Color, 'cornflowerblue').int_rgb


def test_color_from_hex(benchmark):
    """Parse a hex color."""
    assert benchmark(pv.Color, '#4682b4').int_rgb


def test_lookup_table_map_array(rng, benchmark):
    """Map an array of values through a lookup table."""
    lut = pv.LookupTable('viridis')
    values = rng.random(10_000)
    assert benchmark(lut, values).shape == (10_000, 4)


def test_get_cmap_safe(benchmark):
    """Look up a named matplotlib colormap."""
    assert benchmark(get_cmap_safe, 'viridis').N
