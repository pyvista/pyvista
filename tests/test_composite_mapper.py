import pytest

import pyvista as pv
from pyvista.plotting.composite_mapper import CompositePolyDataMapper


@pytest.fixture()
def composite_mapper(multiblock_poly):
    pl = pv.Plotter()
    actor, mapper = pl.add_composite(multiblock_poly)
    return mapper


@pytest.fixture()
def block_attributes(composite_mapper):
    return composite_mapper.block_attr


@pytest.fixture()
def block_attr(block_attributes):
    return block_attributes[0]


def test_basic_mapper(composite_mapper):
    assert isinstance(composite_mapper, CompositePolyDataMapper)


def test_composite_mapper_non_poly(multiblock_all):
    # should run without raising
    pl = pv.Plotter()
    actor, mapper = pl.add_composite(multiblock_all)


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
    assert block_attr.color == pv.Color(color).float_rgb

    color = (1, 1, 0)
    block_attr.color = color
    assert block_attr.color == pv.Color(color).float_rgb

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


def test_block_attributes(block_attributes):
    color = (1.0, 1.0, 1.0)
    pickable = False
    opacity = 0.5
    visible = False
    block_attributes[0].color = color
    block_attributes[0].pickable = pickable
    block_attributes[0].opacity = opacity
    block_attributes[0].visible = visible

    assert block_attributes[0].color == color
    assert block_attributes[0].pickable == pickable
    assert block_attributes[0].opacity == opacity
    assert block_attributes[0].visible == visible

    block_attributes.reset_colors()
    assert block_attributes[0].color is None

    block_attributes.reset_pickability()
    assert block_attributes[0].pickable is None

    block_attributes.reset_opacities()
    assert block_attributes[0].opacity is None

    block_attributes.reset_visibilities()
    assert block_attributes[0].visible is None


def test_remove_scalars_on_collection(multiblock_poly):
    multi_multi = pv.MultiBlock([multiblock_poly, multiblock_poly])
    pl = pv.Plotter()
    pl.add_composite(multi_multi, scalars='multi-comp')
    assert 'multi-comp-normed' in multiblock_poly[2].point_data
    pl.close()
    assert 'multi-comp-normed' not in multiblock_poly[2].point_data
