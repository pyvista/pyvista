"""Benchmarks for slicing, ufuncs and assignment on pyvista_ndarray."""

from __future__ import annotations

import operator

import numpy as np


def test_ndarray_slice(ndarray, benchmark):
    """Slice a pyvista_ndarray."""
    assert benchmark(operator.itemgetter(slice(10, 5000)), ndarray).size


def test_ndarray_ufunc(ndarray, benchmark):
    """Apply a ufunc to a pyvista_ndarray."""
    assert benchmark(np.multiply, ndarray, 2.0).size


def test_ndarray_setitem(ndarray, benchmark):
    """Assign into a slice of a pyvista_ndarray."""
    benchmark(ndarray.__setitem__, slice(0, 100), 1.0)
    assert ndarray[0] == 1.0
