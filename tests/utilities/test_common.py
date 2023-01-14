""" Tests for pyvista.utilities.common."""

import numpy as np
import pytest

from pyvista.utilities.arrays import _coerce_pointslike_arg


def test_coerce_point_like_arg():
    # Test with Sequence
    point = [1.0, 2.0, 3.0]
    coerced_arg, singular = _coerce_pointslike_arg(point)
    assert isinstance(coerced_arg, np.ndarray)
    assert coerced_arg.shape == (1, 3)
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))
    assert singular

    # Test with 1D np.ndarray
    point = np.array([1.0, 2.0, 3.0])
    coerced_arg, singular = _coerce_pointslike_arg(point)
    assert isinstance(coerced_arg, np.ndarray)
    assert coerced_arg.shape == (1, 3)
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))
    assert singular

    # Test with (1, 3) np.ndarray
    point = np.array([[1.0, 2.0, 3.0]])
    coerced_arg, singular = _coerce_pointslike_arg(point)
    assert isinstance(coerced_arg, np.ndarray)
    assert coerced_arg.shape == (1, 3)
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))
    assert not singular

    # Test with 2D ndarray
    point = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coerced_arg, singular = _coerce_pointslike_arg(point)
    assert isinstance(coerced_arg, np.ndarray)
    assert coerced_arg.shape == (2, 3)
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert not singular


def test_coerce_point_like_arg_copy():
    # Sequence is always copied
    point = [1.0, 2.0, 3.0]
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=True)
    point[0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))

    point = [1.0, 2.0, 3.0]
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=False)
    point[0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))

    # 1D np.ndarray can be copied or not
    point = np.array([1.0, 2.0, 3.0])
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=True)
    point[0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0]]))

    point = np.array([1.0, 2.0, 3.0])
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=False)
    point[0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[10.0, 2.0, 3.0]]))

    # 2D np.ndarray can be copied or not
    point = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=True)
    point[0, 0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    point = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coerced_arg, _ = _coerce_pointslike_arg(point, copy=False)
    point[0, 0] = 10.0
    assert np.array_equal(coerced_arg, np.array([[10.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_coerce_point_like_arg_errors():
    # wrong length sequence
    with pytest.raises(ValueError):
        _coerce_pointslike_arg([1, 2])

    # wrong type
    with pytest.raises(TypeError):
        # allow Sequence but not Iterable
        _coerce_pointslike_arg({1, 2, 3})

    # wrong length ndarray
    with pytest.raises(ValueError):
        _coerce_pointslike_arg(np.empty(4))
    with pytest.raises(ValueError):
        _coerce_pointslike_arg(np.empty([2, 4]))

    # wrong ndim ndarray
    with pytest.raises(ValueError):
        _coerce_pointslike_arg(np.empty([1, 3, 3]))


def test_coerce_points_like_args_does_not_copy():
    source = np.random.rand(100, 3)
    output, _ = _coerce_pointslike_arg(source)  # test that copy=False is default
    output /= 2
    assert np.array_equal(output, source)
    assert np.may_share_memory(output, source)
