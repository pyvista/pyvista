import pytest

import pyvista
from pyvista.plotting.composite_mapper import CompositeMapper


@pytest.fixture()
def composite_mapper(multiblock):
    pl = pyvista.Plotter()
    actor, prop, mapper = pl.add_composite(multiblock)
    return mapper


@pytest.fixture()
def block_attributes(composite_mapper):
    return composite_mapper.block_attr


@pytest.fixture()
def block_attr(block_attributes):
    return block_attributes[0]


def test_basic_mapper(composite_mapper):
    assert isinstance(composite_mapper, CompositeMapper)


def test_block_attr(block_attributes):
    with pytest.raises(KeyError, match='Invalid block key'):
        block_attributes[-1]
    with pytest.raises(KeyError, match='out of bounds'):
        block_attributes[100000]
    with pytest.raises(TypeError, match='got float'):
        block_attributes[0.5]


def test_visible(block_attr):
    assert block_attr.visible is None

    visible = True
    block_attr.visible = visible
    assert block_attr.visible is visible

    block_attr.visible = None
    assert block_attr.visible is None


def test_opacity(block_attr):
    # unset must be none
    assert block_attr.opacity is None

    opacity = 0.5
    block_attr.opacity = opacity
    assert block_attr.opacity == opacity

    block_attr.opacity = None
    assert block_attr.opacity is None


def test_color(block_attr):
    # when unset this must be None
    assert block_attr.color is None

    color = 'red'
    block_attr.color = color
    assert block_attr.color == pyvista.Color(color).float_rgb

    color = (1, 1, 0)
    block_attr.color = color
    assert block_attr.color == pyvista.Color(color).float_rgb

    block_attr.color = None
    assert block_attr.color is None


def test_attr_repr(block_attr):
    block_attr.visible = True
    block_attr.opacity = 0.9
    block_attr.color = 'blue'
    block_attr.pickable = False

    repr_ = repr(block_attr)
    assert 'True' in repr_
    assert 'False' in repr_
    assert '1.0' in repr_
    assert '0.9' in repr_
