from __future__ import annotations

import re

import pytest

import pyvista as pv
from pyvista.plotting.assembly import MultiProperty


@pytest.fixture()
def multi_property_green():
    return MultiProperty([pv.Property(color='green')])


@pytest.fixture()
def multi_property_blue_and_red():
    return MultiProperty([pv.Property(color='red'), pv.Property(color='blue')])


def test_multi_property_set_get_common_prop_value(
    multi_property_green, multi_property_blue_and_red
):
    assert multi_property_green.color == 'green'
    assert multi_property_green._props[0].color == 'green'

    match = (
        'Multiple Property values detected for attribute "color". Value can only be returned if all Property objects have the same value.\n'
        'Got: `Color(name=\'blue\', hex=\'#0000ffff\', opacity=255)` and `Color(name=\'red\', hex=\'#ff0000ff\', opacity=255)`.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        _ = multi_property_blue_and_red.color
    multi_property_blue_and_red.color = 'purple'
    assert multi_property_blue_and_red.color == 'purple'
    assert multi_property_blue_and_red._props[0].color == 'purple'
    assert multi_property_blue_and_red._props[1].color == 'purple'


def test_multi_property_repr_matches_property():
    prop = pv.Property(color='red')
    multi_prop = MultiProperty([prop])

    prop_repr_lines = repr(prop).splitlines()[1:]
    multi_prop_repr_lines = repr(multi_prop).splitlines()[1:]
    assert prop_repr_lines == multi_prop_repr_lines


def test_multi_property_repr_multiple_values(multi_property_blue_and_red):
    repr_lines = repr(multi_property_blue_and_red).splitlines()[1:]
    # Non-exhaustive test, only check a few properties are as expected
    assert '  Color:                       MULTIPLE VALUES' in repr_lines
    assert '  Opacity:                     1.0' in repr_lines
