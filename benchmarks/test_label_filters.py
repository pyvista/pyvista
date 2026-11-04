"""Benchmarks for the array-scanning label and value filters."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize('boundary_style', ['external', 'strict_external'])
def test_contour_labels(label_image, benchmark, boundary_style):
    """Contour external boundaries in an image with many internal label boundaries."""
    contours = benchmark(
        label_image.contour_labels,
        boundary_style=boundary_style,
        scalars='labels',
        simplify_output=False,
    )
    assert contours.n_cells
    assert np.isfinite(contours.points).all()
    labels = contours.cell_data['boundary_labels']
    assert labels.shape == (contours.n_cells, 2)
    assert np.all(labels[:, 0] > 0)
    assert np.all(labels[:, 1] == 0)


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
