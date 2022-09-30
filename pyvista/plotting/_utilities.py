"""Utilities for plotting."""
from typing import Tuple

import numpy as np


def view_vectors(view: str, negative: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Given a plane to view, return vectors for setting up camera.

    Parameters
    ----------
    view : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy'}
        Plane to return vectors for.
    negative : bool
        Whether to view from opposite direction.  Default ``False``.

    Returns
    -------
    direction: List
        [x, y, z] vector that points in the viewing direction
    viewup: List
        [x, y, z] vector that points to the viewup direction

    """
    if view == 'xy':
        vec = np.array([0, 0, 1])
        viewup = np.array([0, 1, 0])
    elif view == 'yx':
        vec = np.array([0, 0, -1])
        viewup = np.array([1, 0, 0])
    elif view == 'xz':
        vec = np.array([0, -1, 0])
        viewup = np.array([0, 0, 1])
    elif view == 'zx':
        vec = np.array([0, 1, 0])
        viewup = np.array([1, 0, 0])
    elif view == 'yz':
        vec = np.array([1, 0, 0])
        viewup = np.array([0, 0, 1])
    elif view == 'zy':
        vec = np.array([-1, 0, 0])
        viewup = np.array([0, 1, 0])
    else:
        raise ValueError(
            f"Unexpected value for direction {view}\n"
            "    Expected: 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'"
        )

    if negative:
        vec *= -1
    return vec, viewup
