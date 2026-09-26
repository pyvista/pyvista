"""Typing cases for :class:`pyvista.AffineWidget3D`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def on_matrix(matrix: NDArray[np.float64]) -> None:
    """Accept the actor's float64 user matrix."""


pl = pv.Plotter()
actor = pl.add_mesh(pv.Sphere())
assert_types(
    pv.AffineWidget3D(pl, actor, release_callback=on_matrix, interact_callback=on_matrix),
    pv.AffineWidget3D,
)
