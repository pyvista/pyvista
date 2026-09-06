"""Typing cases for :func:`pyvista.core.utilities.arrays.parse_field_choice`."""

from __future__ import annotations

from typing import Literal

from type_assert import assert_types

from pyvista.core.utilities.arrays import FieldAssociation
from pyvista.core.utilities.arrays import parse_field_choice


def as_association() -> FieldAssociation:
    """Return an association typed only as the enum."""
    return FieldAssociation.POINT


assert_types(parse_field_choice('point'), Literal[FieldAssociation.POINT])
assert_types(parse_field_choice('p'), Literal[FieldAssociation.POINT])
assert_types(parse_field_choice('points'), Literal[FieldAssociation.POINT])
assert_types(parse_field_choice(FieldAssociation.POINT), Literal[FieldAssociation.POINT])

assert_types(parse_field_choice('cell'), Literal[FieldAssociation.CELL])
assert_types(parse_field_choice('c'), Literal[FieldAssociation.CELL])
assert_types(parse_field_choice('cells'), Literal[FieldAssociation.CELL])
assert_types(parse_field_choice(FieldAssociation.CELL), Literal[FieldAssociation.CELL])

assert_types(parse_field_choice('field'), Literal[FieldAssociation.NONE])
assert_types(parse_field_choice('f'), Literal[FieldAssociation.NONE])
assert_types(parse_field_choice('fields'), Literal[FieldAssociation.NONE])
assert_types(parse_field_choice(FieldAssociation.NONE), Literal[FieldAssociation.NONE])

assert_types(parse_field_choice('row'), Literal[FieldAssociation.ROW])
assert_types(parse_field_choice('r'), Literal[FieldAssociation.ROW])
assert_types(parse_field_choice(FieldAssociation.ROW), Literal[FieldAssociation.ROW])

# The catch-all, reached only by an argument widened to the enum
assert_types(parse_field_choice(as_association()), FieldAssociation)
