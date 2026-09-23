"""Benchmarks for the array-scanning label and value filters."""

from __future__ import annotations


def test_color_labels_index(label_image, benchmark):
    """Color labels by using their values as colormap indices."""
    colored = benchmark(label_image.color_labels, coloring_mode='index')
    assert colored.cell_data


def test_color_labels_cycle(label_image, benchmark):
    """Color labels by cycling through the colormap."""
    colored = benchmark(label_image.color_labels, coloring_mode='cycle')
    assert colored.cell_data


def test_select_values(value_image, benchmark):
    """Select a range of values from an image."""
    assert benchmark(value_image.select_values, ranges=[10, 20]).n_points
