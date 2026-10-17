"""Typing cases for :attr:`pyvista.core.utilities.writer.PLYWriter.texture`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

from pyvista.core.utilities.writer import PLYWriter
from tests.typing.meshes import poly

if TYPE_CHECKING:
    from numpy.typing import NDArray

MESH = poly()


def uint8_colors() -> NDArray[np.uint8]:
    """Return one RGB colour per point of the mesh."""
    return np.zeros((MESH.n_points, 3), dtype=np.uint8)


def a_writer() -> PLYWriter:
    """Return a PLY writer for the mesh."""
    return PLYWriter('mesh.ply', MESH)


def with_texture(texture: str | NDArray[np.uint8] | None) -> PLYWriter:
    """Return a PLY writer with ``texture`` set after the colours are stored."""
    writer = a_writer()
    writer.texture = uint8_colors()
    writer.texture = texture
    return writer


assert_types(with_texture(uint8_colors()).texture, str | None)
assert_types(with_texture('_color_array').texture, str | None)
assert_types(with_texture(None).texture, str | None)
