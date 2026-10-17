"""Benchmarks for the input validators every public filter calls."""

from __future__ import annotations

import numpy as np
import pyvista_validation as pvv


def test_validate_array3(benchmark):
    """Validate a length-three vector."""
    assert benchmark(pvv.validate_array3, (1.0, 2.0, 3.0)).shape == (3,)


def test_validate_arrayN(benchmark):  # noqa: N802
    """Validate a one-dimensional array."""
    values = np.arange(1000)
    assert benchmark(pvv.validate_arrayN, values).size


def test_validate_arrayNx3(dense_points, benchmark):  # noqa: N802
    """Validate an array of points."""
    points = dense_points[:1000]
    assert benchmark(pvv.validate_arrayNx3, points).shape == (1000, 3)


def test_validate_axes(benchmark):
    """Validate an orthonormal axes matrix."""
    assert benchmark(pvv.validate_axes, np.eye(3)).shape == (3, 3)


def test_validate_transform4x4(benchmark):
    """Validate a four-by-four transform."""
    assert benchmark(pvv.validate_transform4x4, np.eye(4)).shape == (4, 4)


def test_validate_data_range(benchmark):
    """Validate a two-element data range."""
    assert benchmark(pvv.validate_data_range, (0.0, 1.0))[1] == 1.0
