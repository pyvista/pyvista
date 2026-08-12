"""Unit tests for pyvista.ext._autoenum internals, without a full Sphinx build."""

from __future__ import annotations

from enum import Enum
from enum import Flag
from enum import IntEnum

import pytest

from pyvista.ext import _autoenum as autoenum


def test_is_enum():
    class Color(Enum):
        RED = 1

    assert autoenum._is_enum(Color)
    assert not autoenum._is_enum(int)
    assert not autoenum._is_enum(Color.RED)  # an instance, not the class itself


def test_metaclass_properties_finds_only_metaclass_properties():
    class Meta(type):
        @property
        def computed(cls):
            return 'value'

    class Plain(metaclass=Meta):
        @property
        def instance_only(self):
            return 'nope'

    props = autoenum._metaclass_properties(Plain)
    assert set(props) == {'computed'}
    assert props['computed'] is Meta.__dict__['computed']


def test_metaclass_properties_empty_for_plain_enum():
    class Color(Enum):
        RED = 1

    assert autoenum._metaclass_properties(Color) == {}


def test_metaclass_properties_skips_enummeta_itself():
    class Color(Enum):
        RED = 1

    assert autoenum._metaclass_properties(type(Color)) == {}


def test_metaclass_property_names_sorted():
    assert autoenum.metaclass_property_names('enum', 'IntFlag') == []


def test_metaclass_property_names_on_celltype():
    assert 'dimension_map' in autoenum.metaclass_property_names('pyvista', 'CellType')


def test_metaclass_properties_first_definition_wins_in_mro():
    class BaseMeta(type):
        @property
        def shared(cls):
            return 'base'

    class SubMeta(BaseMeta):
        @property
        def shared(self):
            return 'sub'

    class Widget(metaclass=SubMeta): ...

    props = autoenum._metaclass_properties(Widget)
    assert props['shared'] is SubMeta.__dict__['shared']


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([0, 1, 2, 4, 8], True),
        ([0], True),
        ([1, 2, 3], False),  # 3 is not a power of two
        ([5, 9, 22], False),  # arbitrary VTK-style type codes, like CellType
    ],
)
def test_is_bitmask_like(values, expected):
    Bits = IntEnum('Bits', {f'V{i}': v for i, v in enumerate(values)})
    assert autoenum._is_bitmask_like(Bits) is expected


def test_is_bitmask_like_true_for_flag_regardless_of_values():
    class NotReallyBits(Flag):
        A = 3  # not a power of two, but Flag membership alone is enough to opt in

    assert autoenum._is_bitmask_like(NotReallyBits)


@pytest.mark.parametrize(
    ('value', 'as_hex', 'expected'),
    [
        (0, False, '0'),
        (42, False, '42'),
        (0, True, '0x0'),
        (255, True, '0xff'),
    ],
)
def test_format_value(value, as_hex, expected):
    assert autoenum._format_value(value, as_hex=as_hex) == expected
