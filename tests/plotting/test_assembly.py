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


def test_multi_property_repr(multi_property_green, multi_property_blue_and_red):
    actual_lines = repr(multi_property_green).splitlines()[1:]
    assert actual_lines == [
        '  Style:                       "Surface"',
        "  Color:                       Color(name='green', hex='#008000ff', opacity=255)",
        "  Edge color:                  Color(name='black', hex='#000000ff', opacity=255)",
        '  Opacity:                     1.0',
        '  Edge opacity:                1.0',
        '  Show edges:                  False',
        '  Lighting:                    True',
        '  Ambient:                     0.0',
        '  Diffuse:                     1.0',
        '  Specular:                    0.0',
        '  Specular power:              100.0',
        '  Metallic:                    0.0',
        '  Roughness:                   0.5',
        '  Interpolation:               0',
        '  Render points as spheres:    False',
        '  Render lines as tubes:       False',
        '  Line width:                  1.0',
        '  Point size:                  5.0',
        '  Culling:                     "none"',
        "  Ambient color:               Color(name='green', hex='#008000ff', opacity=255)",
        "  Specular color:              Color(name='green', hex='#008000ff', opacity=255)",
        "  Diffuse color:               Color(name='green', hex='#008000ff', opacity=255)",
        '  Anisotropy:                  0.0',
    ]

    actual_lines = repr(multi_property_blue_and_red).splitlines()[1:]
    assert actual_lines == [
        '  Style:                       "Surface"',
        '  Color:                       MULTIPLE VALUES',
        "  Edge color:                  Color(name='black', hex='#000000ff', opacity=255)",
        '  Opacity:                     1.0',
        '  Edge opacity:                1.0',
        '  Show edges:                  False',
        '  Lighting:                    True',
        '  Ambient:                     0.0',
        '  Diffuse:                     1.0',
        '  Specular:                    0.0',
        '  Specular power:              100.0',
        '  Metallic:                    0.0',
        '  Roughness:                   0.5',
        '  Interpolation:               0',
        '  Render points as spheres:    False',
        '  Render lines as tubes:       False',
        '  Line width:                  1.0',
        '  Point size:                  5.0',
        '  Culling:                     "none"',
        '  Ambient color:               MULTIPLE VALUES',
        '  Specular color:              MULTIPLE VALUES',
        '  Diffuse color:               MULTIPLE VALUES',
        '  Anisotropy:                  0.0',
    ]
