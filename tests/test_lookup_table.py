import numpy as np
import pytest

from pyvista import Color, LookupTable


@pytest.fixture
def lut():
    return LookupTable()


@pytest.fixture
def lut_w_cmap():
    return LookupTable('viridis')


def test_values(lut):
    values = np.array([[0, 0, 0, 0], [255, 255, 255, 255]])
    lut.values = values
    assert lut.values.dtype == np.uint8
    assert np.allclose(lut.values, values)


def test_apply_cmap(lut):
    n_values = 5
    lut.cmap = 'reds'
    lut.n_values = n_values
    assert lut.values.shape == (n_values, 4)
    assert lut.n_values == n_values


def test_cmap_init():
    new_lut = LookupTable('gray', n_values=2, flip=True)
    assert np.allclose([[254, 255, 255, 255], [0, 0, 0, 255]], new_lut.values)


def test_annotations(lut):
    assert lut.annotations == {}
    anno = {0: 'low', 0.5: 'medium', 1: 'high'}
    lut.annotations = anno
    assert lut.annotations == anno


def test_value_range(lut, lut_w_cmap):
    assert lut_w_cmap.value_range is None

    value_range = (0, 1.0)
    lut.value_range = value_range
    assert lut.value_range == value_range


def test_hue_range(lut, lut_w_cmap):
    assert lut_w_cmap.hue_range is None

    hue_range = (0, 1.0)
    lut.hue_range = hue_range
    assert lut.hue_range == hue_range


def test_saturation_range(lut, lut_w_cmap):
    assert lut_w_cmap.saturation_range is None

    saturation_range = (0, 1.0)
    lut.saturation_range = saturation_range
    assert lut.saturation_range == saturation_range


def test_alpha_range(lut, lut_w_cmap):
    assert lut_w_cmap.alpha_range is None

    alpha_range = (0, 1.0)
    lut.alpha_range = alpha_range
    assert lut.alpha_range == alpha_range


def test_nan_color(lut):
    lut.nan_color = 'b'
    assert lut.nan_color == Color('b')


def test_below_range_color(lut):
    lut.below_range_color = 'r'
    assert lut.below_range_color == Color('r')
    assert lut.GetUseBelowRangeColor()

    lut.below_range_color = None
    assert lut.below_range_color is None
    assert not lut.GetUseBelowRangeColor()


def test_above_range_color(lut):
    lut.above_range_color = 'r'
    assert lut.above_range_color == Color('r')
    assert lut.GetUseAboveRangeColor()

    lut.above_range_color = None
    assert lut.above_range_color is None
    assert not lut.GetUseAboveRangeColor()


def test_ramp(lut):
    lut.ramp = 'linear'
    assert lut.ramp == 'linear'
    with pytest.raises(ValueError, match='must be one of the following'):
        lut.ramp = 'foo'


def test_log_scale(lut):
    lut.log_scale = True
    assert lut.log_scale is True

    lut.log_scale = False
    assert lut.log_scale is False


def test_repr(lut):
    assert 'VTK' in repr(lut)

    lut.values = lut.values**0.5
    assert 'From values' in repr(lut)

    lut.cmap = 'viridis'
    assert 'viridis' in repr(lut)


def test_table_range(lut):
    table_range = (0.5, 1.0)
    lut.table_range = table_range
    assert lut.table_range == table_range


def test_table_cmap_list(lut):
    lut.cmap = ['reds', 'greens', 'blues']
    assert lut.n_values == 3
