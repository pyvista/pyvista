"""Common functions."""
import collections
from typing import Sequence, Union

import numpy as np


def _coerce_points_like_arg(points: Union[Sequence, np.ndarray], copy: bool=True):
    """Check and coerce arg to (n, 3) np.ndarray.

    Parameters
    ----------
    points : Sequence(float) or np.ndarray
        Argument to coerce into (n, 3) ``np.ndarray``.
    copy : bool
        Whether to copy the ``points`` array.  Copying always occurs if ``points``
        is not ``np.ndarray``.

    Returns
    -------
    np.ndarray : size (n, 3) array

    """
    if isinstance(points, collections.abc.Sequence):
            points = np.asarray(points)

    if not isinstance(points, np.ndarray):
        raise TypeError("Given points must be a sequence or an array.")

    if points.ndim > 2:
        raise ValueError("Array of points must be 1D or 2D")
    if points.ndim == 2:
        if points.shape[1] != 3:
            raise ValueError("Array of points must have three values per point"
                            "(shape (n, 3))")
    else:
        if points.size != 3:
            raise ValueError("Given point must have three values")
        points = np.reshape(points, [1, 3])

    if copy:
        return points.copy()
    return points
