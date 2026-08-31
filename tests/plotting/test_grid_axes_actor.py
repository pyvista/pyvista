"""Test the GridAxesActor wrapping."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting.grid_axes_actor import GRID_AXES_MIN_VTK_VERSION

pytestmark = pytest.mark.needs_vtk_version(
    GRID_AXES_MIN_VTK_VERSION, reason='vtkGridAxesActor3D was added in VTK 9.5'
)

UNIT_CUBE = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)


@pytest.fixture
def grid_axes_actor():
    return pv.GridAxesActor(bounds=UNIT_CUBE)


class RecordingGridAxesActor(pv.GridAxesActor):
    """Record the tick positions pushed into VTK."""

    def __init__(self, **kwargs):
        self.counts = []
        self.labels = []
        super().__init__(**kwargs)

    def SetNumberOfLabels(self, axis, count):  # noqa: N802
        """Record and forward."""
        self.counts.append((axis, count))
        super().SetNumberOfLabels(axis, count)

    def SetLabel(self, axis, index, value):  # noqa: N802
        """Record and forward."""
        self.labels.append((axis, index, value))
        super().SetLabel(axis, index, value)


def test_default_bounds():
    assert pv.GridAxesActor().bounds == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def test_bounds(grid_axes_actor):
    grid_axes_actor.bounds = (-1, 2, -3, 4, -5, 6)
    assert grid_axes_actor.bounds == (-1, 2, -3, 4, -5, 6)
    assert grid_axes_actor.center == (0.5, 0.5, 0.5)


def test_update_bounds(grid_axes_actor):
    grid_axes_actor.update_bounds((0, 2, 0, 2, 0, 2))
    assert grid_axes_actor.bounds == (0, 2, 0, 2, 0, 2)


def test_axes_ranges():
    actor = pv.GridAxesActor(bounds=UNIT_CUBE, axes_ranges=(0, 100, 0, 200, 0, 300))
    # The grid is placed at the bounds but labelled with the ranges
    assert np.allclose(actor.bounds, UNIT_CUBE)
    assert actor.GetGridBounds() == (0, 100, 0, 200, 0, 300)


def test_padding():
    actor = pv.GridAxesActor(bounds=(0, 10, 0, 10, 0, 10), padding=0.1)
    assert actor.bounds == (-1, 11, -1, 11, -1, 11)


@pytest.mark.parametrize('padding', [1.5, -0.1, 'a'])
def test_padding_raises(padding):
    with pytest.raises(ValueError, match='padding'):
        pv.GridAxesActor(padding=padding)


def test_bounds_raises():
    with pytest.raises(ValueError, match=r'has shape \(5,\) which is not allowed'):
        pv.GridAxesActor(bounds=(0, 1, 0, 1, 0))


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_axis_visibility(axis):
    index = 'xyz'.index(axis)
    actor = pv.GridAxesActor(bounds=UNIT_CUBE, **{f'{axis}_axis_visibility': False})
    assert getattr(actor, f'{axis}_axis_visibility') is False
    assert actor.GetTitle(index) == ''

    setattr(actor, f'{axis}_axis_visibility', True)
    assert actor.GetTitle(index) == f'{axis.upper()} Axis'


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_label_visibility(axis):
    index = 'xyz'.index(axis)
    actor = RecordingGridAxesActor(bounds=UNIT_CUBE, **{f'{axis}_label_visibility': False})
    assert getattr(actor, f'{axis}_label_visibility') is False
    # The title survives even though the labels are gone
    assert actor.GetTitle(index) == f'{axis.upper()} Axis'
    # The hidden axis is the only one given no tick positions
    assert actor.counts == [(index, 0)]


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_titles(axis):
    actor = pv.GridAxesActor(bounds=UNIT_CUBE, **{f'{axis}_title': 'foo'})
    assert getattr(actor, f'{axis}_title') == 'foo'
    assert actor.GetTitle('xyz'.index(axis)) == 'foo'

    setattr(actor, f'{axis}_title', 'bar')
    assert getattr(actor, f'{axis}_title') == 'bar'


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_title_must_be_string(axis):
    with pytest.raises(TypeError, match=r'must be an instance of .*str'):
        pv.GridAxesActor(**{f'{axis}_title': None})


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_n_labels(axis):
    index = 'xyz'.index(axis)
    actor = RecordingGridAxesActor(bounds=UNIT_CUBE, **{f'n_{axis}labels': 3})
    assert getattr(actor, f'n_{axis}labels') == 3
    assert actor.counts == [(index, 3)]
    # Three positions spanning the unit cube
    assert actor.labels == [(index, 0, 0.0), (index, 1, 0.5), (index, 2, 1.0)]

    actor.counts.clear()
    setattr(actor, f'n_{axis}labels', None)
    assert getattr(actor, f'n_{axis}labels') is None
    # Automatic labelling places no custom positions on any axis
    assert actor.counts == []


@pytest.mark.parametrize(('fmt', 'precision'), [('%.3f', 3), ('{0:.2f}', 2), ('{:.1f}', 1)])
def test_label_format(fmt, precision):
    actor = pv.GridAxesActor(bounds=UNIT_CUBE, x_label_format=fmt)
    assert actor.x_label_format == fmt
    assert actor.GetPrecision(0) == precision


def test_label_format_default_is_automatic(grid_axes_actor):
    assert grid_axes_actor.x_label_format is None


@pytest.mark.parametrize('fmt', ['{0:.1f} m', 'Value', '%d'])
def test_label_format_raises(fmt):
    with pytest.raises(ValueError, match='is not supported by GridAxesActor'):
        pv.GridAxesActor(x_label_format=fmt)


def test_color_default(grid_axes_actor):
    expected = pv.Color(pv.global_theme.font.color).float_rgb
    assert grid_axes_actor.GetProperty().GetColor() == expected
    assert grid_axes_actor.GetTitleTextProperty(0).GetColor() == expected
    assert grid_axes_actor.GetLabelTextProperty(0).GetColor() == expected


def test_color():
    actor = pv.GridAxesActor(bounds=UNIT_CUBE, color='red')
    expected = pv.Color('red').float_rgb
    # SetProperty applies to every face, not just the first
    assert actor.GetProperty().GetColor() == expected
    assert actor.GetTitleTextProperty(2).GetColor() == expected


def test_font():
    actor = pv.GridAxesActor(bounds=UNIT_CUBE, font_size=42, font_family='times', bold=False)
    prop = actor.GetTitleTextProperty(0)
    assert prop.GetFontSize() == 42
    assert prop.GetFontFamilyAsString() == 'Times'
    assert not prop.GetBold()


def test_font_defaults(grid_axes_actor):
    prop = grid_axes_actor.GetLabelTextProperty(1)
    assert prop.GetFontSize() == pv.global_theme.font.size
    assert prop.GetBold()


@pytest.mark.parametrize('value', [True, False])
def test_grid(value):
    assert pv.GridAxesActor(grid=value).grid is value


@pytest.mark.parametrize('value', [True, False])
def test_ticks(value):
    assert pv.GridAxesActor(ticks=value).ticks is value


@pytest.mark.parametrize('value', [True, False])
def test_unique_edges_only(value):
    assert pv.GridAxesActor(unique_edges_only=value).unique_edges_only is value


def test_label_offset():
    actor = pv.GridAxesActor(label_offset=(3, 4))
    assert actor.label_offset == (3, 4)
    actor.label_offset = (5, 6)
    assert actor.label_offset == (5, 6)


def test_no_camera_required():
    # Unlike CubeAxesActor the actor renders without being given a camera
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.add_actor(pv.GridAxesActor(bounds=UNIT_CUBE))
    pl.show()


def _title_ink(title):
    """Return the number of dark pixels drawn for a single titled axis."""
    pl = pv.Plotter(window_size=(400, 300))
    pl.set_background('white')
    pl.show_bounds(
        bounds=UNIT_CUBE,
        xtitle=title,
        ytitle='',
        ztitle='',
        show_xlabels=False,
        show_ylabels=False,
        show_zlabels=False,
        show_yaxis=False,
        show_zaxis=False,
        color='black',
        actor='grid',
    )
    pl.camera_position = 'xy'
    pl.show(auto_close=False)
    image = pl.screenshot(return_img=True)
    pl.close()
    return int((image.mean(axis=2) < 128).sum())


@pytest.mark.skipif(not pv.check_math_text_support(), reason='Math text is not supported')
def test_math_text_title_is_rendered_as_symbols():
    """Math text in a title renders as a symbol rather than as its source string.

    Covers https://github.com/pyvista/pyvista/issues/8691, where the cube actor drew
    the literal characters ``$\\rho$``.
    """
    # Subtract the grid itself so only the title glyphs are counted
    grid_only = _title_ink('')
    math_text = _title_ink(r'$\rho$') - grid_only
    six_characters = _title_ink('ABCDEF') - grid_only
    one_character = _title_ink('A') - grid_only

    assert six_characters > 0
    # A symbol was drawn, rather than the title being dropped altogether
    assert math_text > 0
    # A single rendered glyph uses far less ink than the six characters of its source
    assert math_text < six_characters / 2
    assert math_text < one_character * 3
