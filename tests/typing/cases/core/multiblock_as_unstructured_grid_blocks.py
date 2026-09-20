"""Typing cases for :meth:`pyvista.MultiBlock.as_unstructured_grid_blocks`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import ImageData
from pyvista import MultiBlock
from pyvista import UnstructuredGrid
from pyvista.core.composite import _NestedUnstructuredGrid


def multi() -> MultiBlock:
    """Return a `MultiBlock` with no declared block type."""
    return pv.MultiBlock([pv.PolyData()])


def images() -> MultiBlock[ImageData]:
    """Return a `MultiBlock` declared to hold only `ImageData`."""
    return pv.MultiBlock([pv.ImageData()])


def optional_images() -> MultiBlock[ImageData | None]:
    """Return a `MultiBlock` declared to hold `ImageData` or missing blocks."""
    return pv.MultiBlock([pv.ImageData(), None])


# A declared block type cannot nest, so every block of the output is an `UnstructuredGrid`
assert_types(images().as_unstructured_grid_blocks(), MultiBlock[UnstructuredGrid])
assert_types(optional_images().as_unstructured_grid_blocks(), MultiBlock[UnstructuredGrid])

# An undeclared block type may nest, and a nested block stays nested
assert_types(multi().as_unstructured_grid_blocks(), MultiBlock[_NestedUnstructuredGrid])
