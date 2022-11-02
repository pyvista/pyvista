"""Internal array utilities."""
import collections.abc
from typing import Tuple, Union

import numpy as np

from pyvista._typing import NumericArray, VectorArray


def _coerce_pointslike_arg(
    points: Union[NumericArray, VectorArray], copy: bool = False
) -> Tuple[np.ndarray, bool]:
    """Check and coerce arg to (n, 3) np.ndarray.

    Parameters
    ----------
    points : Sequence(float) or np.ndarray
        Argument to coerce into (n, 3) ``np.ndarray``.

    copy : bool, default: False
        Whether to copy the ``points`` array.  Copying always occurs if ``points``
        is not :class:`numpy.ndarray`.

    Returns
    -------
    numpy.ndarray
        Size (n, 3) array.
    bool
        Whether the input was a single point in an array-like with shape (3,).

    """
    if isinstance(points, collections.abc.Sequence):
        points = np.asarray(points)

    if not isinstance(points, np.ndarray):
        raise TypeError("Given points must be a sequence or an array.")

    if points.ndim > 2:
        raise ValueError("Array of points must be 1D or 2D")

    if points.ndim == 2:
        if points.shape[1] != 3:
            raise ValueError("Array of points must have three values per point (shape (n, 3))")
        singular = False

    else:
        if points.size != 3:
            raise ValueError("Given point must have three values")
        singular = True
        points = np.reshape(points, [1, 3])

    if copy:
        return points.copy(), singular
    return points, singular
