"""Typing cases for :meth:`pyvista.MultiBlock.as_polydata_blocks`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import ImageData
from pyvista import MultiBlock
from pyvista import PolyData
from pyvista.core.composite import _NestedPolyData


def multi() -> MultiBlock:
    """Return a `MultiBlock` with no declared block type."""
    return pv.MultiBlock([pv.PolyData()])


def images() -> MultiBlock[ImageData]:
    """Return a `MultiBlock` declared to hold only `ImageData`."""
    return pv.MultiBlock([pv.ImageData()])


def optional_images() -> MultiBlock[ImageData | None]:
    """Return a `MultiBlock` declared to hold `ImageData` or missing blocks."""
    return pv.MultiBlock([pv.ImageData(), None])


# A declared block type cannot nest, so every block of the output is a `PolyData`
assert_types(images().as_polydata_blocks(), MultiBlock[PolyData])
assert_types(optional_images().as_polydata_blocks(), MultiBlock[PolyData])

# An undeclared block type may nest, and a nested block stays nested
assert_types(multi().as_polydata_blocks(), MultiBlock[_NestedPolyData])
