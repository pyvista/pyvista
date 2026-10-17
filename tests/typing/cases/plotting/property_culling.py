"""Typing cases for :attr:`pyvista.Property.culling`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from pyvista.plotting._typing import BackfaceArgs


def a_backface_dict() -> BackfaceArgs:
    """Return backface parameters naming culling by one of its aliases."""
    return {'culling': 'backface', 'style': 'points'}


assert_types(pv.Property(culling='back').culling, str)
assert_types(pv.Property(culling='front').culling, str)
assert_types(pv.Property(culling='none').culling, str)

# The aliases and bools `add_mesh` documents
assert_types(pv.Property(culling='backface').culling, str)
assert_types(pv.Property(culling='b').culling, str)
assert_types(pv.Property(culling='f').culling, str)
assert_types(pv.Property(culling=True).culling, str)
assert_types(pv.Property(culling=False).culling, str)

# `backface_params` unpacks into `Property`, so every `BackfaceArgs` value must reach it
assert_types(pv.Property(**a_backface_dict()).culling, str)
