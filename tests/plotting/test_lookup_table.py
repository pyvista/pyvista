from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyvista as pv
from pyvista import Color
from pyvista import LookupTable
from pyvista.plotting import _vtk

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def lut():
    return LookupTable()


@pytest.fixture
def lut_w_cmap():
    return LookupTable('viridis')


def test_cmap_values_raises():
    with pytest.raises(
        ValueError,
        match=re.escape('Cannot set both `cmap` and `values`.'),
    ):
        LookupTable(cmap='foo', values='bar')


def test_call_raises(lut: LookupError, mocker: MockerFixture):
    from pyvista.plotting import lookup_table

    m = mocker.patch.object(lookup_table, 'np')
    m.array.side_effect = TypeError

    with pytest.raises(
        TypeError,
        match=re.escape('LookupTable __call__ expects a single value or an iterable.'),
    ):
        lut('foo')


def test_values(lut):
    values = [
        [0, 0, 0, 255],
        [85, 0, 0, 255],
        [170, 0, 0, 255],
        [255, 0, 0, 255],
    ]
    lut.values = values
    assert lut.values.dtype == np.uint8
    assert np.allclose(lut.values, values)

    with pytest.raises(RuntimeError, match='cannot be set'):
        lut.n_values = 10


def test_apply_cmap(lut):
    n_values = 5
    lut.cmap = 'reds'
    lut.n_values = n_values
    assert lut.values.shape == (n_values, 4)
    assert lut.n_values == n_values


def test_init_cmap():
    new_lut = LookupTable('gray', n_values=2, flip=True)
    assert np.allclose([[254, 255, 255, 255], [0, 0, 0, 255]], new_lut.values)


def test_init_values():
    new_lut = LookupTable(values=[[254, 255, 255, 255], [0, 0, 0, 255]])
    assert np.allclose([[254, 255, 255, 255], [0, 0, 0, 255]], new_lut.values)


def test_init_custom():
    value_range = (0.1, 0.2)
    hue_range = (0.2, 0.3)
    alpha_range = (0.3, 0.4)
    scalar_range = (1, 9)
    ramp = 'linear'
    nan_color = Color('r')
    above_range_color = Color('b')
    below_range_color = Color('g')
    log_scale = True
    annotations = {0: 'low', 4.5: 'medium', 9: 'high'}

    new_lut = LookupTable(
        value_range=value_range,
        hue_range=hue_range,
        alpha_range=alpha_range,
        scalar_range=scalar_range,
        ramp=ramp,
        nan_color=nan_color,
        above_range_color=above_range_color,
        below_range_color=below_range_color,
        log_scale=log_scale,
        annotations=annotations,
    )

    assert new_lut.value_range == value_range
    assert new_lut.hue_range == hue_range
    assert new_lut.alpha_range == alpha_range
    assert new_lut.scalar_range == scalar_range
    assert new_lut.ramp == ramp
    assert new_lut.nan_color == nan_color
    assert new_lut.above_range_color == above_range_color
    assert new_lut.below_range_color == below_range_color
    assert new_lut.log_scale == log_scale
    assert new_lut.annotations == annotations


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

    lut.below_range_color = True
    assert lut.below_range_color == pv.global_theme.below_range_color


def test_above_range_color(lut):
    lut.above_range_color = 'r'
    assert lut.above_range_color == Color('r')
    assert lut.GetUseAboveRangeColor()

    lut.above_range_color = None
    assert lut.above_range_color is None
    assert not lut.GetUseAboveRangeColor()

    lut.above_range_color = True
    assert lut.above_range_color == pv.global_theme.above_range_color


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
    assert 'PyVista' in repr(lut)

    lut.values = lut.values**0.5
    assert 'From values' in repr(lut)

    lut.cmap = 'viridis'
    assert 'viridis' in repr(lut)

    # try a colorcet
    lut.cmap = 'fire'
    assert 'fire' in repr(lut)


def test_scalar_range(lut):
    scalar_range = (0.5, 1.0)
    lut.scalar_range = scalar_range
    assert lut.scalar_range == scalar_range


def test_table_cmap_list(lut):
    lut.cmap = ['red', 'green', 'blue']
    assert lut.n_values == 3


@pytest.mark.skip_check_gc
def test_table_values_update(lut):
    lut.cmap = 'Greens'
    lut.values[:, -1] = np.linspace(0, 255, lut.n_values)
    assert lut.values.max() == 255
    assert lut.values[:, 2].max() < 255


def test_to_tf(lut):
    tf = lut.to_color_tf()
    assert isinstance(tf, _vtk.vtkColorTransferFunction)


def test_map_value(lut):
    assert lut.map_value(0.5) == (0.0, 1.0, 0.0, 1.0)


def test_call(lut):
    n_values = 10
    arr = lut(np.linspace(0, 1, n_values))
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == n_values

    assert lut.map_value(0.5) == lut.map_value(0.5)


@pytest.mark.skip_check_gc
def test_custom_opacity(lut):
    values_copy = lut.values.copy()
    lut.apply_opacity('sigmoid')
    assert not np.array_equiv(lut.values[:, -1], 255)
    # check RGB isn't changed when applying an opacity
    assert np.array_equal(values_copy[:, :-1], lut.values[:, :-1])

    # ensure opacity is not reset when changing the colormap
    opac_orig = lut.values[:, -1].copy()
    lut.cmap = 'jet'
    assert np.array_equal(lut.values[:, -1], opac_orig)

    lut.apply_opacity(0)
    assert np.array_equiv(lut.values[:, -1], 0)
    lut.apply_opacity(0.5)
    assert np.array_equiv(lut.values[:, -1], int(255 * 0.5))
    lut.apply_opacity(1)
    assert np.array_equiv(lut.values[:, -1], 255)

    with pytest.raises(ValueError, match='between 0 and 1'):
        lut.apply_opacity(-0.1)


@pytest.mark.parametrize('clamping', [True, False])
def test_to_opacity_tf(lut, clamping):
    tf = lut.to_opacity_tf(clamping=clamping)
    assert isinstance(tf, _vtk.vtkPiecewiseFunction)
    assert tf.GetClamping() == int(clamping)
