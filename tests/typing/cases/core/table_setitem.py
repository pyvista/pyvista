"""Typing cases for :meth:`pyvista.Table.__setitem__`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def table() -> pv.Table:
    """Return a table with three rows."""
    return pv.Table({'a': np.zeros(3)})


assert_types(table().__setitem__('f', [0.0, 1.0, 2.0]), None)
assert_types(table().__setitem__('v', [[0.0, 0.0, 0.0]] * 3), None)
assert_types(table().__setitem__('s', ['x'] * 3), None)
