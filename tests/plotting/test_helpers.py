"""Tests for plotting helpers."""

import numpy as np
import pytest

from pyvista.plotting.helpers import view_vectors


def test_view_vectors():
    views = ('xy', 'yx', 'xz', 'zx', 'yz', 'zy')

    for view in views:
        vec, viewup = view_vectors(view)
        assert isinstance(vec, np.ndarray)
        assert np.array_equal(vec.shape, (3,))
        assert isinstance(viewup, np.ndarray)
        assert np.array_equal(viewup.shape, (3,))

    with pytest.raises(ValueError, match="Unexpected value for direction"):
        view_vectors('invalid')
