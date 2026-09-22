from __future__ import annotations

import itertools
import math
import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.core.errors import VTKVersionError
from pyvista.plotting.scalar_bars import _bar_title_height
from pyvista.plotting.scalar_bars import _box_pixels
from pyvista.plotting.scalar_bars import _fitted_label_font
from pyvista.plotting.scalar_bars import _fitting_font
from pyvista.plotting.scalar_bars import _label_size
from pyvista.plotting.scalar_bars import _label_texts
from pyvista.plotting.scalar_bars import _label_ticks
from pyvista.plotting.scalar_bars import _lifted_ramp
from pyvista.plotting.scalar_bars import _nudge
from pyvista.plotting.scalar_bars import _ramp_room
from pyvista.plotting.scalar_bars import _text_size
from pyvista.plotting.scalar_bars import _title_height
from pyvista.plotting.scalar_bars import _title_width

KEY = 'Data'
WIDE_KEY = 'Pressure (Pa)'
WIDE_FONT = 22
# Four-digit labels, too wide for a bar on a small window to hold apart at that size
WIDE_RANGE = (0.0, 6932.0)
# Enough labels to crowd a bar on a small window at the size they ask for
CROWDED_LABELS = 9
LARGE_WINDOW = (1024, 768)
HALF_WINDOW = tuple(size // 2 for size in LARGE_WINDOW)
THIRD_WINDOW = tuple(size // 3 for size in LARGE_WINDOW)
SMALL_WINDOW = (400, 300)


@pytest.fixture
def scalar_bars(sphere):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    pl.add_scalar_bar(
        KEY,
        interactive=True,
        vertical=False,
        outline=True,
        fmt='%10.5f',
        nan_annotation=True,
        fill=True,
        background_color='k',
    )
    return pl.scalar_bars


def test_repr(scalar_bars):
    repr_ = repr(scalar_bars)
    assert f'"{KEY}"' in repr_
    assert 'False' in repr_ or 'True' in repr_, 'missing interactive flag'
    assert 'Scalar Bar Title     Interactive' in repr_


def test_remove_fail(scalar_bars):
    scalar_bars.add_scalar_bar('MOARDATA', mapper=scalar_bars._plotter.mapper)
    with pytest.raises(ValueError, match='Multiple scalar bars'):
        scalar_bars.remove_scalar_bar()


def test_add_fail(scalar_bars):
    with pytest.raises(ValueError, match="Exactly one of 'mapper'"):
        scalar_bars.add_scalar_bar('MOARDATA')


def test_dict(scalar_bars):
    assert KEY in scalar_bars
    assert 'Data' in scalar_bars.keys()
    assert len(scalar_bars) == 1
    assert next(iter(scalar_bars.keys())) == KEY
    assert isinstance(next(iter(scalar_bars.values())), _vtk.vtkScalarBarActor)

    for key, value in scalar_bars.items():
        assert isinstance(value, _vtk.vtkScalarBarActor)
        assert key == 'Data'

    assert isinstance(scalar_bars[KEY], _vtk.vtkScalarBarActor)


def test_clear(scalar_bars):
    assert len(scalar_bars) == 1
    scalar_bars.clear()
    assert len(scalar_bars) == 0


def test_update_title(scalar_bars):
    new_title = 'Elevation'
    scalar_bars.update_title(KEY, new_title)

    # Verify internal dicts are re-keyed
    assert KEY not in scalar_bars
    assert new_title in scalar_bars
    assert new_title in scalar_bars._scalar_bar_ranges
    assert new_title in scalar_bars._scalar_bar_mappers
    assert len(scalar_bars) == 1

    # Verify VTK actor title is updated
    assert scalar_bars[new_title].GetTitle() == new_title

    # Verify slot lookup is re-keyed
    assert KEY not in scalar_bars._plotter._scalar_bar_slot_lookup
    assert new_title in scalar_bars._plotter._scalar_bar_slot_lookup


def test_update_title_render(scalar_bars):
    scalar_bars.update_title(KEY, 'NewTitle', render=True)
    assert 'NewTitle' in scalar_bars


def test_update_title_no_slot(sphere):
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    pl.add_scalar_bar(KEY, mapper=pl.mapper, position_x=0.2, position_y=0.2)
    pl.scalar_bars.update_title(KEY, 'NewTitle')
    assert 'NewTitle' in pl.scalar_bars


def test_update_title_same(scalar_bars):
    scalar_bars.update_title(KEY, KEY)
    assert KEY in scalar_bars
    assert len(scalar_bars) == 1


def test_update_title_not_found(scalar_bars):
    with pytest.raises(KeyError, match='not found'):
        scalar_bars.update_title('DoesNotExist', 'New')


def test_update_title_conflict(scalar_bars):
    scalar_bars.add_scalar_bar('Other', mapper=scalar_bars._plotter.mapper)
    with pytest.raises(ValueError, match='already exists'):
        scalar_bars.update_title(KEY, 'Other')


def test_update_title_image(sphere, verify_image_cache):
    verify_image_cache.windows_skip_image_cache = True

    sphere['Data'] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars='Data')
    pl.scalar_bars.update_title('Data', 'Elevation')
    pl.show()


@pytest.mark.usefixtures('verify_image_cache')
def test_background_color_fill(sphere):
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    scalar_bar = pl.add_scalar_bar(
        KEY,
        background_color='gray',
        fill=True,
        outline=True,
        width=0.8,
        height=0.3,
        position_x=0.1,
        position_y=0.05,
        label_font_size=40,
        title_font_size=40,
    )
    assert scalar_bar.GetBackgroundProperty().GetColor() == pytest.approx(
        pv.Color('gray').float_rgb
    )
    pl.show()


@pytest.mark.usefixtures('verify_image_cache')
def test_background_color_keeps_out_of_range_colors(sphere):
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    actor = pl.add_mesh(
        sphere,
        clim=[-0.2, 0.2],
        below_color='magenta',
        above_color='red',
        scalar_bar_args={
            'background_color': 'cyan',
            'fill': True,
            'width': 0.8,
            'height': 0.3,
            'position_x': 0.1,
            'position_y': 0.05,
            'label_font_size': 40,
            'title_font_size': 40,
        },
    )
    lut = pl.scalar_bar.GetLookupTable()
    assert lut is actor.mapper.lookup_table
    assert lut.below_range_color == pv.Color('magenta')
    assert lut.above_range_color == pv.Color('red')
    pl.show()


@pytest.mark.usefixtures('verify_image_cache')
def test_background_color_composite_range(multiblock_poly):
    pl = pv.Plotter()
    pl.add_composite(
        multiblock_poly,
        scalars='data_a',
        clim=[0.2, 10],
        scalar_bar_args={
            'background_color': 'white',
            'color': 'black',
            'fill': True,
            'outline': True,
            'width': 0.8,
            'height': 0.3,
            'position_x': 0.1,
            'position_y': 0.05,
            'label_font_size': 40,
            'title_font_size': 40,
        },
    )
    assert pl.scalar_bar.GetLookupTable().GetRange() == (0.2, 10.0)
    pl.update_scalar_bar_range([1, 5])
    assert pl.scalar_bar.GetLookupTable().GetRange() == (1.0, 5.0)
    pl.show()


@pytest.mark.usefixtures('verify_image_cache')
def test_labels_centered_with_translucent_actor(sphere):
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(
        sphere,
        scalar_bar_args={
            'vertical': False,
            'n_labels': 3,
            'width': 0.8,
            'height': 0.3,
            'position_x': 0.1,
            'position_y': 0.05,
            'label_font_size': 40,
            'title_font_size': 40,
        },
    )
    # The translucent actor makes VTK lay out the labels from a stale justification
    pl.add_mesh(pv.Cube(center=(2, 0, 0)), opacity=0.5)
    pl.show()


def test_ticks_off(sphere):
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars=sphere.points[:, 2], show_scalar_bar=False)
    bar = pl.add_scalar_bar(tick_locations=[-0.4, 0.0, 0.4], n_labels=0)
    assert not bar.GetDrawTickLabels()


@pytest.mark.usefixtures('verify_image_cache')
def test_ticks(sphere):
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    scalar_bar = pl.add_scalar_bar(
        KEY,
        tick_locations=[-100, 25, 110],
        fmt='%.0f',
        width=0.8,
        height=0.3,
        position_x=0.1,
        position_y=0.05,
        label_font_size=40,
        title_font_size=40,
    )
    assert scalar_bar.GetUseCustomLabels()
    assert list(pv.convert_array(scalar_bar.GetCustomLabels())) == [-100.0, 25.0, 110.0]
    pl.show()


@pytest.mark.usefixtures('verify_image_cache')
def test_categories_label_positions():
    # Every label sits over the middle of its own color block
    values = [0.0, 1.0, 2.0, 5.0, 6.0, 9.0]
    mesh = pv.ImageData(dimensions=(len(values), 2, 2))
    mesh[KEY] = np.tile(values, 4)
    pl = pv.Plotter()
    actor = pl.add_mesh(
        mesh,
        categories=True,
        cmap='glasbey',
        scalar_bar_args={
            'width': 0.9,
            'height': 0.3,
            'position_x': 0.05,
            'position_y': 0.05,
            'label_font_size': 40,
            'title_font_size': 40,
        },
    )
    actor.visibility = False
    assert list(pv.convert_array(pl.scalar_bars[KEY].GetCustomLabels())) == values
    pl.show()


@pytest.mark.usefixtures('verify_image_cache')
def test_categories_label_positions_clim():
    # An explicit range halves the end blocks and draws their labels at the bar ends
    values = [0.0, 1.0, 2.0, 5.0, 6.0, 9.0]
    mesh = pv.ImageData(dimensions=(len(values), 2, 2))
    mesh[KEY] = np.tile(values, 4)
    pl = pv.Plotter()
    actor = pl.add_mesh(
        mesh,
        categories=True,
        cmap='glasbey',
        clim=(values[0], values[-1]),
        scalar_bar_args={
            'width': 0.9,
            'height': 0.3,
            'position_x': 0.05,
            'position_y': 0.05,
            'label_font_size': 40,
            'title_font_size': 40,
        },
    )
    actor.visibility = False
    assert pl.mapper.lookup_table.scalar_range == (values[0], values[-1])
    pl.show()


def test_too_many_scalar_bars():
    pl = pv.Plotter()
    with pytest.raises(RuntimeError, match='Maximum number of color'):  # noqa: PT012
        for i in range(100):  # pragma: no branch -- raises before the loop ends
            mesh = pv.Sphere()
            mesh[str(i)] = range(mesh.n_points)
            pl.add_mesh(mesh)


@pytest.mark.parametrize('unique_bar', [True, False])
@pytest.mark.parametrize('shape', [(1, 1), (2, 2), (3, 3)])
def test_unique_scalar_bars(sphere, unique_bar: bool, shape: tuple[int, int]):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(shape=shape)
    pairs = list(itertools.product(range(shape[0]), range(shape[1])))
    for i, j in pairs:
        pl.subplot(i, j)
        pl.add_mesh(sphere, show_scalar_bar=True, scalar_bar_args={'unique_bar': unique_bar})

    key_scalar_bars = [b for b in pl.scalar_bars.values() if b.GetTitle() == KEY]

    if unique_bar:
        assert len(key_scalar_bars) == shape[0] * shape[1]
    else:
        assert len(key_scalar_bars) == 1


@pytest.mark.parametrize('vertical', [True, False])
@pytest.mark.parametrize('title_pad', [0.0, 0.25, 0.5, 1.0])
def test_title_pad(sphere, vertical: bool, title_pad: float):
    sphere[KEY] = sphere.points[:, 2]
    font_size = 20

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    pl.add_scalar_bar(
        KEY,
        vertical=vertical,
        title_font_size=font_size,
        label_font_size=font_size,
        title_pad=title_pad,
    )

    offset = pl.scalar_bar.GetTitleTextProperty().GetLineOffset()
    assert offset == -round(title_pad * font_size)


@pytest.mark.parametrize('vertical', [True, False])
def test_title_pad_from_theme(sphere, vertical: bool):
    sphere[KEY] = sphere.points[:, 2]
    font_size = 20
    config = 'colorbar_vertical' if vertical else 'colorbar_horizontal'
    getattr(pv.global_theme, config).title_pad = 0.75

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    pl.add_scalar_bar(KEY, vertical=vertical, title_font_size=font_size, label_font_size=font_size)

    assert pl.scalar_bar.GetTitleTextProperty().GetLineOffset() == -15


def test_title_pad_constrained_font_size(sphere):
    # A constrained font is re-fit to its box, so the padding is not applied
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    pl.add_scalar_bar(KEY, title_pad=0.5)

    scalar_bar = pl.scalar_bar
    assert not scalar_bar.GetUnconstrainedFontSize()
    assert scalar_bar.GetTitleTextProperty().GetLineOffset() == 0


@pytest.mark.parametrize(('outline', 'fill'), [(True, False), (False, True)])
def test_title_pad_boxed(sphere, outline: bool, fill: bool):
    # A box grows to hold the padded title, so the padding survives the box
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    pl.add_scalar_bar(
        KEY,
        title_font_size=20,
        label_font_size=20,
        title_pad=0.5,
        outline=outline,
        fill=fill,
        background_color='grey',
    )

    assert _title_gap(pl, pl.scalar_bar) >= 10


@pytest.mark.parametrize(
    'title', ['Range {i}', 'A wide range {i}'], ids=['short_title', 'wide_title']
)
@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
@pytest.mark.usefixtures('verify_image_cache')
def test_stacked_bars_render(sphere, vertical: bool, title: str):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    # Move the vertical bars off the window edge so a wide title is not clipped by it
    pl.theme.colorbar_vertical.position_x = 0.8
    pl.add_mesh(sphere, show_scalar_bar=False)
    for i in range(3):
        pl.add_scalar_bar(
            title.format(i=i),
            vertical=vertical,
            title_font_size=14,
            label_font_size=14,
            n_labels=3,
            mapper=pl.mapper,
        )
    pl.show()


@pytest.mark.parametrize('window_size', [[400, 300], [600, 400], [1024, 768]])
def test_stacked_horizontal_bars_clear_their_annotations(sphere, window_size):
    # The pitch is a fraction of the window, so a small one must still fit the text
    sphere[KEY] = sphere.points[:, 2]
    font_size = 18

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bars = [
        pl.add_scalar_bar(
            f'{KEY}{i}',
            vertical=False,
            title_font_size=font_size,
            label_font_size=font_size,
            mapper=pl.mapper,
        )
        for i in range(3)
    ]

    height = pl.theme.colorbar_horizontal.height
    needed = bars[0].GetBarRatio() * height * window_size[1] + 2 * font_size
    pitches = [
        (b.GetPosition()[1] - a.GetPosition()[1]) * window_size[1]
        for a, b in itertools.pairwise(bars)
    ]
    assert pitches[0] == pytest.approx(pitches[1])
    assert min(pitches) >= needed


def test_stacked_vertical_bars_clear_their_titles(sphere):
    # Titles are centred on the bar and are wider than it, so they set the pitch
    sphere[KEY] = sphere.points[:, 2]
    window_size = [600, 400]
    title = 'A wide scalar bar title'

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bars = [
        pl.add_scalar_bar(f'{title}{i}', vertical=True, title_font_size=18, mapper=pl.mapper)
        for i in range(3)
    ]

    dpi = pl.render_window.GetDPI()
    widest = max(_title_width(b.GetTitleTextProperty(), b.GetTitle(), dpi) for b in bars)
    pitches = [
        (a.GetPosition()[0] - b.GetPosition()[0]) * window_size[0]
        for a, b in itertools.pairwise(bars)
    ]
    gap = 0.2 * pl.theme.colorbar_vertical.width * window_size[0]
    assert min(pitches) == pytest.approx(widest + gap)


@pytest.mark.parametrize(
    ('titles', 'kwargs'),
    [
        (['Short', 'A very much longer title', 'A bit long'], {}),
        (['Short', 'Super duper long title'], {'width': 0.2}),
    ],
    ids=['uneven_titles', 'wide_bars'],
)
def test_stacked_vertical_bars_clear_an_uneven_neighbor(sphere, titles, kwargs):
    # A title is centered on its bar, so it must stop short of the neighboring ramp
    sphere[KEY] = sphere.points[:, 2]
    window_size = [900, 400]

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bars = [
        pl.add_scalar_bar(title, vertical=True, title_font_size=18, mapper=pl.mapper, **kwargs)
        for title in titles
    ]

    dpi = pl.render_window.GetDPI()
    for bar, neighbor in _stacked_pairs(bars):
        title = _title_width(bar.GetTitleTextProperty(), bar.GetTitle(), dpi)
        center = (bar.GetPosition()[0] + bar.GetWidth() / 2) * window_size[0]
        assert center + title / 2 < neighbor.GetPosition()[0] * window_size[0]
    pl.close()


def _label_reach(bar, dpi, window_width):
    """Return the x of the far edge of a bar's tick labels, in pixels."""
    ramp = bar.GetBarRatio() * bar.GetWidth() * window_width
    return (
        bar.GetPosition()[0] * window_width
        + ramp
        + _label_size(bar, bar.GetLabelTextProperty(), dpi)[0]
    )


def _stacked_pairs(bars):
    """Pair each stacked bar with the neighbor it was placed against."""
    return zip(bars[1:], bars[:-1], strict=True)


def _wide_number_bars(plotter, sphere, **kwargs):
    """Stack three bars, the middle one labelled with numbers far wider than its bar."""
    bars = []
    for i, clim in enumerate([(0, 1), (-1234.5, 1234.5), (0, 1)]):
        lut = pv.LookupTable(cmap='viridis', scalar_range=clim)
        mapper = pv.DataSetMapper(sphere)
        mapper.lookup_table = lut
        bars.append(
            plotter.add_scalar_bar(f'{KEY}{i}', vertical=True, n_labels=3, mapper=mapper, **kwargs)
        )
    return bars


def test_stacked_vertical_bars_clear_their_labels(sphere):
    # Labels face the neighboring bar, so the gap fits them whatever the stacking is
    sphere[KEY] = sphere.points[:, 2]
    window_size = [700, 450]

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bars = _wide_number_bars(pl, sphere, label_font_size=22)

    dpi = pl.render_window.GetDPI()
    for bar, neighbor in _stacked_pairs(bars):
        assert _label_reach(bar, dpi, window_size[0]) < neighbor.GetPosition()[0] * window_size[0]
    pl.close()


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_rotated_title_clears_the_neighbors_labels(sphere):
    # A rotated title hangs into the same gap that the next bar's labels use
    sphere[KEY] = sphere.points[:, 2]
    window_size = [700, 450]

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    label_font = 7
    bars = _wide_number_bars(
        pl, sphere, rotate_title=True, title_font_size=40, label_font_size=label_font
    )

    dpi = pl.render_window.GetDPI()
    pad = round(pl.theme.colorbar_vertical.title_pad * label_font)
    for bar, neighbor in _stacked_pairs(bars):
        title = neighbor.GetPosition()[0] * window_size[0] - pad - _bar_title_height(neighbor, dpi)
        assert _label_reach(bar, dpi, window_size[0]) < title
    pl.close()


def _bar_rect(bar, window_size):
    """Return the box a scalar bar draws, as left, right, bottom, top in pixels."""
    window_width, window_height = window_size
    x, y = bar.GetPosition()
    return (
        x * window_width,
        (x + bar.GetWidth()) * window_width,
        y * window_height,
        (y + bar.GetHeight()) * window_height,
    )


def _overlap(first, second):
    """Return the area two scalar bar boxes share, in square pixels."""
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    bottom = max(first[2], second[2])
    top = min(first[3], second[3])
    return max(right - left, 0) * max(top - bottom, 0)


@pytest.mark.parametrize('vertical', [True, False], ids=['vertical_first', 'horizontal_first'])
def test_stacked_bars_clear_a_neighbor_turned_across_them(sphere, vertical: bool):
    # A bar stacks along the axis its neighbor fills, so one turned across that neighbor
    # is cleared in the other direction rather than pushed along the window
    sphere[KEY] = sphere.points[:, 2]
    window_size = [1024, 768]

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bars = [
        pl.add_scalar_bar(
            f'Bar {index}',
            vertical=turned,
            title_font_size=14,
            label_font_size=14,
            mapper=pl.mapper,
        )
        for index, turned in enumerate([vertical, not vertical, vertical])
    ]

    rects = [_bar_rect(bar, window_size) for bar in bars]
    for first, second in itertools.combinations(rects, 2):
        assert _overlap(first, second) == 0


def test_stacked_interactive_bar_keeps_the_place_it_was_given(sphere):
    # An interactive bar is drawn from its widget's representation, which is built before
    # the bar is stacked and would otherwise put it back where it started
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    pl.add_scalar_bar('First', vertical=True, title_font_size=14, mapper=pl.mapper)
    bar = pl.add_scalar_bar(
        'Second', vertical=True, title_font_size=14, interactive=True, mapper=pl.mapper
    )
    stacked = bar.GetPosition()

    pl.screenshot(return_img=True)

    assert bar.GetPosition() == pytest.approx(stacked)


STACKED_TITLES = ['A bit long', 'Short', 'Super duper long']


@pytest.mark.parametrize(
    'layout',
    [
        {},
        {'stacking_gap': 0.25},
        pytest.param(
            {'rotate_title': True},
            marks=pytest.mark.needs_vtk_version(
                9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0'
            ),
        ),
    ],
    ids=['default', 'gap', 'rotated'],
)
@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
@pytest.mark.usefixtures('verify_image_cache')
def test_stacking_layouts_render(sphere, vertical: bool, layout):
    if layout.get('rotate_title') and not vertical:
        pytest.skip('A rotated title is vertical only')
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    # Move the bars off the window edge so a wide title is not clipped by it
    pl.theme.colorbar_vertical.position_x = 0.75
    pl.add_mesh(sphere, show_scalar_bar=False)
    for title in STACKED_TITLES:
        pl.add_scalar_bar(
            title,
            vertical=vertical,
            title_font_size=14,
            label_font_size=14,
            n_labels=3,
            mapper=pl.mapper,
            **layout,
        )
    pl.show()


BOXES = [{'outline': True}, {'fill': True, 'background_color': 'grey'}]
BOX_IDS = ['outline', 'fill']
FIT_TITLE = 'Elevation (m)'


def _box_edges(bar, window_size):
    """Return the pixel edges of the box a scalar bar draws, as left, right, bottom, top."""
    window_width, window_height = window_size
    x, y = bar.GetPosition()
    return (
        x * window_width,
        (x + bar.GetWidth()) * window_width,
        y * window_height,
        (y + bar.GetHeight()) * window_height,
    )


def _laid_out(pl, bar, title):
    """Lay a horizontal boxed bar out the way VTK does, returning its fonts and title gap."""
    viewport = pl.renderer
    width, height = _box_pixels(bar, viewport)
    text_pad = bar.GetTextPad()
    thickness = math.ceil(height * bar.GetBarRatio())
    ramp = int(thickness - _nudge(thickness, text_pad))
    _, lift = _lifted_ramp(ramp, text_pad)
    title_text = bar.GetTitleTextProperty()
    title_font = _fitting_font(
        lambda size: _text_size(viewport, title_text, title, font_size=size),
        width - 2 * text_pad,
        int((height - ramp - lift - text_pad) * bar.GetTitleRatio()),
        start=title_text.GetFontSize(),
    )
    title_box = math.ceil(_text_size(viewport, title_text, title, font_size=title_font)[1])
    labels = _label_texts(bar)
    label_text = bar.GetLabelTextProperty()

    def size_of(size):
        sizes = [_text_size(viewport, label_text, text, font_size=size) for text in labels]
        return max(w for w, _ in sizes), max(h for _, h in sizes)

    label_font = _fitting_font(
        size_of,
        int((_ramp_room(bar, width, ramp) - text_pad * (len(labels) - 1)) / len(labels)),
        height - ramp - 4 * text_pad - title_box,
        start=label_text.GetFontSize(),
    )
    gap = 2 * text_pad - int(bar.GetFrameProperty().GetLineWidth()) - lift
    return title_font, label_font, gap


def _ramp_rect(pl, bar):
    """Return the ramp VTK laid out for a horizontal boxed bar, relative to its box."""
    rect = [0, 0, 0, 0]
    bar.GetScalarBarRect(rect, pl.renderer)
    left, bottom = bar.GetPositionCoordinate().GetComputedViewportValue(pl.renderer)
    return rect[0] - left, rect[1] - bottom, rect[2], rect[3]


def _modelled_ramp(pl, bar, title):
    """Return the ramp the layout model expects VTK to draw, relative to the box."""
    viewport = pl.renderer
    width, height = _box_pixels(bar, viewport)
    text_pad = bar.GetTextPad()
    thickness = math.ceil(height * bar.GetBarRatio())
    ramp = int(thickness - _nudge(thickness, text_pad))
    _, lift = _lifted_ramp(ramp, text_pad)
    label_text = bar.GetLabelTextProperty()
    label_font = _laid_out(pl, bar, title)[1]
    widest = int(
        max(
            _text_size(viewport, label_text, text, font_size=label_font)[0]
            for text in _label_texts(bar)
        )
    )
    return widest // 2, lift, _ramp_room(bar, width, ramp) - widest, ramp


def _title_gap(pl, bar):
    """Return the pixels a horizontal boxed bar leaves between its labels and its title."""
    return _laid_out(pl, bar, bar.GetTitle())[2]


def _blue_text(image):
    """Return the pixels of a render that hold the blue its text is drawn in."""
    red, green, blue = (image[..., channel].astype(int) for channel in range(3))
    # The text is near full blue with little red or green, which the ramp's blues are not
    return (blue > 200) & (red < 150) & (green < 150)


def _colored(image):
    """Return the pixels of a render whose channels are far enough apart to be the ramp."""
    # White, grey and black hold their channels level, and the ramp's colours do not
    return (image.max(axis=2).astype(int) - image.min(axis=2)) > 40


def _ink_bands(pl, bar):
    """Return the heights of the runs of rows holding blue text across a bar's box, bottom up."""
    text = _blue_text(pl.screenshot(return_img=True))
    left, _ = bar.GetPositionCoordinate().GetComputedViewportValue(pl.renderer)
    width = _box_pixels(bar, pl.renderer)[0]
    # The frame's sides run down every row, so they are left out of the columns read
    inked = text[::-1, left + 3 : left + width - 3].any(axis=1)
    bands = [len(list(run)) for holds_ink, run in itertools.groupby(inked) if holds_ink]
    # The frame's top and bottom are a row or two each
    return [band for band in bands if band > 2]


def _text_outside_the_box(pl, bar):
    """Return whether any of a bar's blue text is drawn outside its box."""
    image = pl.screenshot(return_img=True)
    text = _blue_text(image)
    left, bottom = bar.GetPositionCoordinate().GetComputedViewportValue(pl.renderer)
    width, height = _box_pixels(bar, pl.renderer)
    rows = image.shape[0]
    inside = text.copy()
    inside[rows - bottom - height - 2 : rows - bottom + 2, left - 2 : left + width + 2] = False
    assert text.any()
    return bool(inside.any())


def _fitted_bar(plotter, sphere, *, vertical, box, **kwargs):
    """Add one scalar bar that fits its box to its text."""
    return plotter.add_scalar_bar(
        FIT_TITLE,
        vertical=vertical,
        title_font_size=24,
        label_font_size=24,
        n_labels=5,
        mapper=pv.DataSetMapper(sphere),
        **box,
        **kwargs,
    )


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_box_encloses_the_title(sphere, vertical: bool, box):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=vertical, box=box)

    dpi = pl.render_window.GetDPI()
    left, right, bottom, top = _box_edges(bar, pl.window_size)
    title_width = _title_width(bar.GetTitleTextProperty(), FIT_TITLE, dpi)

    if vertical:
        # The title is centered on the box, so the box has to be at least as wide
        assert right - left >= title_width
        assert _label_reach(bar, dpi, pl.window_size[0]) <= right
        # The title is seated back inside the box rather than lifted clear of it
        assert bar.GetTitleTextProperty().GetLineOffset() > 0
    else:
        label_height = _label_size(bar, bar.GetLabelTextProperty(), dpi)[1]
        title_height = _title_height(bar.GetTitleTextProperty(), FIT_TITLE, dpi)
        ramp = bar.GetBarRatio() * (top - bottom)
        assert top - bottom >= ramp + label_height + title_height


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_box_keeps_the_ramp(sphere, vertical: bool, box):
    # The box grows around the ramp rather than taking its size from it
    sphere[KEY] = sphere.points[:, 2]

    def ramp_size(**sizing):
        pl = pv.Plotter()
        pl.add_mesh(sphere, show_scalar_bar=False)
        bar = pl.add_scalar_bar(
            FIT_TITLE,
            vertical=vertical,
            title_font_size=24,
            label_font_size=24,
            n_labels=5,
            mapper=pv.DataSetMapper(sphere),
            **box,
            **sizing,
        )
        window_width, window_height = pl.window_size
        across = bar.GetWidth() * window_width if vertical else bar.GetHeight() * window_height
        size = bar.GetBarRatio() * across
        pl.close()
        return size

    config = pv.global_theme.colorbar_vertical if vertical else pv.global_theme.colorbar_horizontal
    pinned = {'width': config.width, 'height': config.height}
    assert ramp_size() == pytest.approx(ramp_size(**pinned), abs=1.0)


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
def test_fit_box_keeps_the_title_pad(sphere, box):
    # A fitted box grows to hold the padding, so the title is padded after all
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=False, box=box, title_pad=0.5)

    assert _title_gap(pl, bar) >= 12


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
def test_fit_box_holds_the_labels(sphere, box):
    # VTK lays a horizontal box out itself, pulling the ramp in by half a label so the
    # labels at either end are drawn inside the box rather than centered on its edges
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(pl, sphere, vertical=False, box=box, color='blue')

    assert not bar.GetUnconstrainedFontSize()
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.parametrize('asked', [24, 18, 11, 40])
def test_fit_box_keeps_the_font_size(sphere, asked: int):
    # VTK grows each font to the largest that fits the box, so the box is sized to stop
    # it at the size asked for, or one larger where the two measure the same height
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        FIT_TITLE,
        vertical=False,
        outline=True,
        title_font_size=asked,
        label_font_size=asked,
        mapper=pv.DataSetMapper(sphere),
    )

    pl.screenshot(return_img=True)
    title_font, label_font, _ = _laid_out(pl, bar, FIT_TITLE)
    title_text = bar.GetTitleTextProperty()
    heights = [
        _text_size(pl.renderer, title_text, FIT_TITLE, font_size=size)[1]
        for size in (asked, asked + 1)
    ]
    expected = asked + 1 if heights[0] == heights[1] else asked
    assert (title_font, label_font) == (expected, expected)
    assert _modelled_ramp(pl, bar, FIT_TITLE) == _ramp_rect(pl, bar)


@pytest.mark.parametrize(('width', 'padded'), [(0.6, True), (0.3, False)], ids=['wide', 'narrow'])
def test_fit_box_pads_a_given_height(sphere, width: float, padded: bool):
    # A horizontal box keeps the height it is given and spends the spare on padding, as
    # far as the padding leaves the labels the width they need, and on the ramp beyond
    sphere[KEY] = sphere.points[:, 2]

    def laid_out(**size):
        pl = pv.Plotter(window_size=[1024, 768])
        pl.add_mesh(sphere, show_scalar_bar=False)
        bar = _fitted_bar(pl, sphere, vertical=False, box={'outline': True}, width=width, **size)
        pl.screenshot(return_img=True)
        assert bar.GetHeight() == pytest.approx(size.get('height', bar.GetHeight()))
        assert _modelled_ramp(pl, bar, FIT_TITLE) == _ramp_rect(pl, bar)
        fonts = _laid_out(pl, bar, FIT_TITLE)
        pl.close()
        return fonts

    _, free_label_font, _ = laid_out()
    fonts, gaps = set(), []
    for height in (0.12, 0.27, 0.3, 0.305):
        title_font, label_font, gap = laid_out(height=height)
        fonts.add((title_font, label_font))
        gaps.append(gap)
    # The spare never shrinks the text, and a box padded less holds labels no smaller
    assert len(fonts) == 1
    assert title_font == 24
    assert 24 >= label_font >= free_label_font
    assert gaps == sorted(gaps)
    if padded:
        assert gaps[-1] > gaps[0] > 0
    else:
        assert gaps[-1] - gaps[0] < 4


def test_fit_box_shrinks_the_text_to_a_short_box(sphere):
    # A horizontal box given too little height for its text shrinks the text to fit
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(pl, sphere, vertical=False, box={'outline': True}, height=0.05, color='blue')

    assert bar.GetHeight() == pytest.approx(0.05)
    title_font, label_font, _ = _laid_out(pl, bar, FIT_TITLE)
    assert label_font <= title_font < 24
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.parametrize(
    'bare', [{'title': ''}, {'tick_locations': []}], ids=['no_title', 'no_labels']
)
@pytest.mark.parametrize('height', [None, 0.05], ids=['free', 'short'])
def test_fit_box_lays_out_a_bare_bar(sphere, bare, height):
    # A box with nothing to hold on one line still lays the rest out
    sphere[KEY] = sphere.points[:, 2]
    size = {} if height is None else {'height': height}

    pl = pv.Plotter(window_size=[1024, 768])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        vertical=False,
        outline=True,
        title_font_size=24,
        label_font_size=24,
        mapper=pv.DataSetMapper(sphere),
        **size,
        **bare,
    )
    pl.screenshot(return_img=True)

    assert not bar.GetUnconstrainedFontSize()
    assert _box_pixels(bar, pl.renderer)[1] > 0
    if height is not None:
        assert bar.GetHeight() == pytest.approx(height)


def test_fit_box_shrinks_a_wide_title(sphere):
    # A title wider than the box is shrunk to fit it, and the labels keep their size
    sphere[KEY] = sphere.points[:, 2]
    title = 'Elevation above the reference ellipsoid, in metres'

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = pl.add_scalar_bar(
        title,
        vertical=False,
        outline=True,
        width=0.4,
        title_font_size=24,
        label_font_size=20,
        color='blue',
        mapper=pv.DataSetMapper(sphere),
    )

    title_font, label_font, _ = _laid_out(pl, bar, title)
    assert title_font < 24
    assert label_font == 20
    assert not _text_outside_the_box(pl, bar)


def test_fit_box_leaves_room_for_the_swatches(sphere):
    # The swatches for values out of range shorten the ramp the labels share, so the
    # labels are drawn at the size that shorter ramp holds
    sphere[KEY] = sphere.points[:, 2]

    def bands(**kwargs):
        pl = pv.Plotter(window_size=[1024, 768])
        pl.background_color = 'white'
        pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
        bar = pl.add_scalar_bar(
            FIT_TITLE,
            vertical=False,
            width=0.3,
            n_labels=9,
            fmt='%.3f',
            title_font_size=24,
            color='blue',
            mapper=pv.DataSetMapper(sphere),
            **kwargs,
        )
        label_font = _laid_out(pl, bar, FIT_TITLE)[1]
        width = _box_pixels(bar, pl.renderer)[0]
        inked = _ink_bands(pl, bar)
        pl.close()
        return bar, label_font, width, inked

    bar, label_font, width, boxed = bands(
        outline=True, label_font_size=24, below_label='lo', above_label='hi', nan_annotation=True
    )
    assert bar.GetDrawBelowRangeSwatch()
    assert bar.GetDrawAboveRangeSwatch()
    assert _ramp_room(bar, width, 10) < width - 4
    assert label_font < 24
    # Text drawn free at that size is inked the same height as the boxed text
    _, _, _, free = bands(label_font_size=label_font, unconstrained_font_size=True)
    assert boxed[-2:] == free[-2:]


def test_fit_box_follows_a_new_range(sphere):
    # New labels are measured again, so the box holds them at the size they take
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(pl, sphere, vertical=False, box={'outline': True}, width=0.4, color='blue')
    pl.screenshot(return_img=True)
    before = _box_pixels(bar, pl.renderer)[1]

    pl.update_scalar_bar_range([0.0, 1e6], name=FIT_TITLE)
    pl.render()

    assert _label_texts(bar)[-1].endswith('e+06')
    assert _box_pixels(bar, pl.renderer)[1] != before
    assert _modelled_ramp(pl, bar, FIT_TITLE) == _ramp_rect(pl, bar)
    assert not _text_outside_the_box(pl, bar)


def test_fit_box_follows_the_viewport(sphere):
    # The box is a fraction of its viewport, so it is refitted when the viewport changes
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768], shape=(1, 2))
    pl.subplot(0, 1)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=False, box={'outline': True})
    pl.screenshot(return_img=True)
    fonts = _laid_out(pl, bar, FIT_TITLE)[:2]
    height = _box_pixels(bar, pl.renderer)[1]

    pl.renderer.viewport = (0.5, 0.0, 1.0, 0.5)
    pl.render()

    # The box is a larger share of the shorter viewport, and holds its text as before
    assert bar.GetHeight() > height / 768
    assert _laid_out(pl, bar, FIT_TITLE)[:2] == fonts


def test_fit_box_is_taken_over_by_a_bar_added_again(sphere):
    # A bar removed along with its actor and added again is fitted by the new fit alone
    sphere[KEY] = sphere.points[:, 2]

    def boxed_bar(pl, title_pad):
        actor = pl.add_mesh(sphere, show_scalar_bar=False)
        bar = pl.add_scalar_bar(
            FIT_TITLE,
            vertical=False,
            outline=True,
            title_font_size=24,
            label_font_size=24,
            title_pad=title_pad,
            mapper=actor.mapper,
        )
        return actor, bar

    pl = pv.Plotter(window_size=[1024, 768])
    actor, _ = boxed_bar(pl, 0.25)
    pl.remove_actor(actor)
    assert FIT_TITLE not in pl.scalar_bars
    assert not pl.scalar_bars._scalar_bar_fits
    _, bar = boxed_bar(pl, 2.0)
    pl.screenshot(return_img=True)
    pl.render()

    assert len(pl.scalar_bars._scalar_bar_fits) == 1
    assert _title_gap(pl, bar) >= 48


def test_fit_box_lets_the_text_go_without_a_box(sphere):
    # Turning the box off after the fact hands the text its size back, and turning it
    # on lays the text out inside it
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(pl, sphere, vertical=False, box={}, color='blue')
    pl.screenshot(return_img=True)
    assert bar.GetUnconstrainedFontSize()
    assert bar.GetTitleTextProperty().GetLineOffset() == -12

    bar.SetDrawFrame(True)
    pl.render()
    assert not bar.GetUnconstrainedFontSize()
    assert bar.GetTitleTextProperty().GetLineOffset() == 0
    assert _laid_out(pl, bar, FIT_TITLE)[:2] == (24, 24)
    assert not _text_outside_the_box(pl, bar)

    bar.SetDrawFrame(False)
    pl.render()
    assert bar.GetUnconstrainedFontSize()
    assert bar.GetTitleTextProperty().GetLineOffset() == -12


def test_fit_box_leaves_unconstrained_text_its_size(sphere):
    # Text asked to keep its size is not sized to the box, box or no box
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(
        pl, sphere, vertical=False, box={'outline': True}, unconstrained_font_size=True
    )
    pl.screenshot(return_img=True)

    assert bar.GetUnconstrainedFontSize()
    assert bar.GetTitleTextProperty().GetFontSize() == 24
    label_font = bar.GetLabelTextProperty().GetFontSize()
    pad = round(pl.theme.colorbar_horizontal.title_pad * label_font)
    assert bar.GetTitleTextProperty().GetLineOffset() == -pad


@pytest.mark.parametrize(
    ('kwargs', 'log_scale', 'expected'),
    [
        ({'n_labels': 5}, False, ['0.00', '0.25', '0.50', '0.75', '1.00']),
        ({'n_labels': 1}, False, ['0.50']),
        ({'tick_locations': [0.1, 0.9, 2.0]}, False, ['0.10', '0.90', '2.00']),
        ({'n_labels': 3}, True, ['0.01', '0.10', '1.00']),
    ],
    ids=['spaced', 'single', 'custom', 'log'],
)
def test_label_texts(sphere, kwargs, log_scale: bool, expected):
    # The labels are laid out from their text, so that is what is measured
    sphere[KEY] = np.linspace(0.01 if log_scale else 0.0, 1.0, sphere.n_points)

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False, log_scale=log_scale)
    bar = pl.add_scalar_bar(KEY, fmt='%.2f', **kwargs)

    assert _label_texts(bar) == expected


def test_stacked_boxed_bar_reaches_only_to_its_box(sphere):
    # A horizontal box holds its labels, so a bar beside it need only clear the box
    sphere[KEY] = sphere.points[:, 2]

    def second_bar(**box):
        pl = pv.Plotter()
        pl.add_mesh(sphere, show_scalar_bar=False)
        pl.add_scalar_bar('First', vertical=True, title_font_size=14, label_font_size=14)
        bar = pl.add_scalar_bar(
            'Second',
            vertical=False,
            title_font_size=14,
            label_font_size=14,
            mapper=pv.DataSetMapper(sphere),
            **box,
        )
        x = bar.GetPosition()[0]
        pl.close()
        return x

    assert second_bar(outline=True) > second_bar()
    assert second_bar(outline=True, unconstrained_font_size=True) == second_bar()


@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_box_without_a_box(sphere, vertical: bool):
    # There is nothing to fit when no box is drawn
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        FIT_TITLE,
        vertical=vertical,
        title_font_size=24,
        label_font_size=24,
        mapper=pv.DataSetMapper(sphere),
    )

    config = pl.theme.colorbar_vertical if vertical else pl.theme.colorbar_horizontal
    assert bar.GetWidth() == pytest.approx(config.width)
    assert bar.GetHeight() == pytest.approx(config.height)


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
@pytest.mark.parametrize(
    'sizing', [{'width': 0.3}, {'height': 0.3}, {'width': 0.3, 'height': 0.3}]
)
def test_fit_box_keeps_a_given_size(sphere, sizing, box):
    # A box asked for a size of its own is left at that size
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        FIT_TITLE,
        vertical=True,
        title_font_size=24,
        label_font_size=24,
        mapper=pv.DataSetMapper(sphere),
        **box,
        **sizing,
    )

    for name, given in sizing.items():
        assert getattr(bar, f'Get{name.capitalize()}')() == pytest.approx(given)


@pytest.mark.parametrize('window_size', [[400, 300], [1400, 1000]], ids=['small', 'large'])
@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_box_holds_at_any_window_size(sphere, vertical: bool, window_size):
    # The text is measured in pixels while the box is a fraction of the window, so the
    # fit has to be taken from the window it is drawn in
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=vertical, box={'outline': True})

    dpi = pl.render_window.GetDPI()
    left, right, bottom, top = _box_edges(bar, pl.window_size)
    if vertical:
        assert right - left >= _title_width(bar.GetTitleTextProperty(), FIT_TITLE, dpi)
        assert _label_reach(bar, dpi, window_size[0]) <= right
    else:
        label_height = _label_size(bar, bar.GetLabelTextProperty(), dpi)[1]
        title_height = _title_height(bar.GetTitleTextProperty(), FIT_TITLE, dpi)
        assert top - bottom >= bar.GetBarRatio() * (top - bottom) + label_height + title_height


@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_box_hugs_the_text_not_the_size_given(sphere, vertical: bool):
    # The size a bar is given is a fraction of the window while its text is not, so on a
    # large window a box that kept that size would stand well off the text it holds
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    config = pl.theme.colorbar_vertical if vertical else pl.theme.colorbar_horizontal
    generous = 0.5
    if vertical:
        config.width = generous
    else:
        config.height = generous
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=vertical, box={'outline': True})

    assert (bar.GetWidth() if vertical else bar.GetHeight()) < generous


@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_box_refits_a_resized_window(sphere, vertical: bool):
    # The box is a fraction of the window and the text is not, so a narrower window
    # leaves the text outside a box that is not measured again
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[900, 700])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=vertical, box={'outline': True})
    pl.screenshot(return_img=True)

    # A vertical box is squeezed by a narrower window, a horizontal one by a shorter
    pl.window_size = [400, 700] if vertical else [900, 260]
    pl.screenshot(return_img=True)

    dpi = pl.render_window.GetDPI()
    left, right, bottom, top = _box_edges(bar, pl.window_size)
    if vertical:
        assert right - left >= _title_width(bar.GetTitleTextProperty(), FIT_TITLE, dpi)
        assert _label_reach(bar, dpi, pl.window_size[0]) <= right
    else:
        label_height = _label_size(bar, bar.GetLabelTextProperty(), dpi)[1]
        title_height = _title_height(bar.GetTitleTextProperty(), FIT_TITLE, dpi)
        assert top - bottom >= bar.GetBarRatio() * (top - bottom) + label_height + title_height


@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_box_fits_an_interactive_bar(sphere, vertical: bool):
    # An interactive bar is drawn from its widget's representation, so the box the fit
    # measured is the one the representation has to hold
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=vertical, box={'outline': True}, interactive=True)
    pl.screenshot(return_img=True)

    dpi = pl.render_window.GetDPI()
    left, right, bottom, top = _box_edges(bar, pl.window_size)
    label_height = _label_size(bar, bar.GetLabelTextProperty(), dpi)[1]
    title_height = _title_height(bar.GetTitleTextProperty(), FIT_TITLE, dpi)
    if vertical:
        assert right - left >= _title_width(bar.GetTitleTextProperty(), FIT_TITLE, dpi)
        assert _label_reach(bar, dpi, pl.window_size[0]) <= right
    else:
        ramp = bar.GetBarRatio() * (top - bottom)
        assert top - bottom >= ramp + label_height + title_height


@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_box_keeps_a_size_set_on_the_actor(sphere, vertical: bool):
    # The bar is sized after it is added, as LookupTable.plot does, so that is the size
    # it asks for and the one the next fit has to measure against
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[900, 700])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=vertical, box={})
    bar.SetPosition(0.03, 0.1)
    bar.SetPosition2(0.6, 0.7)
    pl.screenshot(return_img=True)

    pl.window_size = [500, 700]
    pl.screenshot(return_img=True)

    assert bar.GetPosition() == pytest.approx((0.03, 0.1))
    assert (bar.GetWidth(), bar.GetHeight()) == pytest.approx((0.6, 0.7))


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
def test_fit_box_fits_a_size_set_on_the_actor(sphere, box):
    # A box around a bar that was sized after it was added is fitted to the text inside
    # that size, not inside the one the bar was added with
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[900, 700])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=True, box=box)
    pl.screenshot(return_img=True)
    fitted_height = bar.GetHeight()

    bar.SetPosition2(bar.GetWidth(), 0.3)
    pl.window_size = [500, 700]
    pl.screenshot(return_img=True)

    assert bar.GetHeight() < fitted_height
    dpi = pl.render_window.GetDPI()
    left, right, _bottom, _top = _box_edges(bar, pl.window_size)
    assert right - left >= _title_width(bar.GetTitleTextProperty(), FIT_TITLE, dpi)
    assert _label_reach(bar, dpi, pl.window_size[0]) <= right


def test_fit_box_follows_a_renamed_bar(sphere):
    # The box is fitted around the title, so renaming the bar has to carry the fit over
    # to the new title and measure it again
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=True, box={'outline': True})
    pl.screenshot(return_img=True)
    width = bar.GetWidth()

    pl.scalar_bars.update_title(FIT_TITLE, 'A very much longer title')
    pl.render()

    assert FIT_TITLE not in pl.scalar_bars._scalar_bar_fits
    assert pl.scalar_bars._scalar_bar_fits['A very much longer title']['title'] == (
        'A very much longer title'
    )
    # The longer title needs a wider box than the one it replaced
    assert bar.GetWidth() > width


@pytest.mark.parametrize('gone', ['_scalar_bar_actors', '_scalar_bar_fits'])
def test_fit_box_ignores_a_bar_that_is_gone(sphere, gone: str):
    # The observer is dropped with the bar, so it only ever fires for a bar it can
    # still measure, and a render that finds neither must pass the fit by
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    _fitted_bar(pl, sphere, vertical=True, box={'outline': True})
    getattr(pl.scalar_bars, gone).pop(FIT_TITLE)

    pl.screenshot(return_img=True)


def test_fit_box_stops_fitting_without_a_render_window(sphere):
    # A closed plotter has no window to drop the observer from
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    _fitted_bar(pl, sphere, vertical=True, box={'outline': True})
    pl.close()

    pl.scalar_bars.clear()

    assert not pl.scalar_bars._scalar_bar_fits


def test_fit_box_stops_fitting_a_removed_bar(sphere):
    # The observer holds the bar, so it has to go when the bar does
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    _fitted_bar(pl, sphere, vertical=True, box={'outline': True})
    assert pl.scalar_bars._scalar_bar_fits

    pl.remove_scalar_bar(FIT_TITLE)

    assert not pl.scalar_bars._scalar_bar_fits


# A baseline is capped at 400 pixels and compared against the render as it is, so a
# window wider than that cannot be image tested
@pytest.mark.parametrize('window_size', [[320, 280], [400, 300]], ids=['narrow', 'wide'])
@pytest.mark.usefixtures('verify_image_cache')
def test_fit_box_window_size_render(sphere, window_size):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    _fitted_bar(pl, sphere, vertical=True, box={'outline': True})
    pl.show()


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
@pytest.mark.usefixtures('verify_image_cache')
def test_fit_box_render(sphere, vertical: bool, box):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    _fitted_bar(pl, sphere, vertical=vertical, box=box)
    pl.show()


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
@pytest.mark.parametrize('title_pad', [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize('font_size', [10, 14, 20, 28])
def test_stacking_rotate_clears_the_bar_by_the_title_pad(sphere, font_size, title_pad):
    # A rotated title clears its bar, and title_pad sets what is left between them
    sphere[KEY] = sphere.points[:, 2]
    window_size = [400, 400]
    width = 0.08

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        KEY,
        vertical=True,
        rotate_title=True,
        title_font_size=font_size,
        label_font_size=font_size,
        title_pad=title_pad,
        width=width,
    )

    assert bar.GetForceVerticalTitle()
    title_text = bar.GetTitleTextProperty()
    # The offset inflates the reported text bounds, so measure without it
    probe = _vtk.vtkTextProperty()
    probe.ShallowCopy(title_text)
    probe.SetLineOffset(0)
    height = _title_height(probe, KEY, pl.render_window.GetDPI())
    # The title moves two pixels for every unit of line offset
    shift = -2 * title_text.GetLineOffset()
    gap = shift - (width * window_size[0] + height) + 4
    assert gap == pytest.approx(round(title_pad * font_size), abs=2)
    pl.close()


def test_stacking_gap_invalid(sphere):
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    with pytest.raises(ValueError, match='stacking_gap'):
        pl.add_scalar_bar(KEY, stacking_gap=-0.1, mapper=pl.mapper)
    pl.close()


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_stacking_gap_overrides_a_rotated_title(sphere):
    # An explicit gap replaces the space a turned title would otherwise claim
    sphere[KEY] = sphere.points[:, 2]
    window_size = [500, 500]

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bars = [
        pl.add_scalar_bar(
            f'{KEY}{i}',
            vertical=True,
            rotate_title=True,
            stacking_gap=0.1,
            title_font_size=20,
            mapper=pl.mapper,
        )
        for i in range(2)
    ]

    assert bars[1].GetTitleTextProperty().GetLineOffset() < 0
    step = (bars[0].GetPosition()[0] - bars[1].GetPosition()[0]) * window_size[0]
    assert step == pytest.approx(0.1 * window_size[0])
    pl.close()


def test_rotate_title_rejects_horizontal_bars(sphere):
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    with pytest.raises(ValueError, match='not supported for horizontal'):
        pl.add_scalar_bar(KEY, vertical=False, rotate_title=True, mapper=pl.mapper)
    pl.close()


def test_rotate_title_needs_vtk_94(sphere, monkeypatch):
    monkeypatch.setattr(pv, 'vtk_version_info', (9, 3, 0))
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    with pytest.raises(VTKVersionError, match=re.escape('requires VTK 9.4.0')):
        pl.add_scalar_bar(KEY, vertical=True, rotate_title=True, mapper=pl.mapper)
    pl.close()


@pytest.mark.parametrize('vertical', [True, False])
def test_stacking_gap_from_theme(sphere, vertical: bool):
    sphere[KEY] = sphere.points[:, 2]
    window_size = [500, 500]
    config = 'colorbar_vertical' if vertical else 'colorbar_horizontal'
    getattr(pv.global_theme, config).stacking_gap = 0.25

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bars = [
        pl.add_scalar_bar(f'{KEY}{i}', vertical=vertical, title_font_size=20, mapper=pl.mapper)
        for i in range(2)
    ]

    axis = 0 if vertical else 1
    step = (bars[1].GetPosition()[axis] - bars[0].GetPosition()[axis]) * window_size[axis]
    assert step == pytest.approx(-0.25 * 500 if vertical else 0.25 * 500)
    pl.close()


def test_stacking_gap_theme_invalid():
    with pytest.raises(ValueError, match='stacking_gap'):
        pv.global_theme.colorbar_vertical.stacking_gap = -0.1


def test_title_pad_keeps_stacked_spacing(sphere):
    # Stacked horizontal bars move up by as much as their title does
    sphere[KEY] = sphere.points[:, 2]
    font_size = 20

    def positions(title_pad):
        pl = pv.Plotter(window_size=[400, 400])
        pl.add_mesh(sphere, show_scalar_bar=False)
        for i in range(3):
            pl.add_scalar_bar(
                f'{KEY}{i}',
                vertical=False,
                title_font_size=font_size,
                label_font_size=font_size,
                title_pad=title_pad,
                mapper=pl.mapper,
            )
        return [bar.GetPosition()[1] for bar in pl.scalar_bars.values()]

    padded = positions(0.5)
    unpadded = positions(0.0)

    assert padded[0] == unpadded[0]
    for slot, (pad_y, plain_y) in enumerate(zip(padded, unpadded, strict=True)):
        assert pad_y == pytest.approx(plain_y + slot * 10 / 400)


def test_add_scalar_bar_shared_range_resync(sphere):
    # Mappers sharing a scalar bar keep one common range as meshes are added
    pl = pv.Plotter()
    low = sphere.copy()
    low['data'] = np.linspace(0, 1, low.n_points)
    mid = sphere.copy()
    mid['data'] = np.linspace(0.2, 0.8, mid.n_points)
    high = sphere.copy()
    high['data'] = np.linspace(-1, 5, high.n_points)
    actors = [pl.add_mesh(mesh, scalars='data') for mesh in (low, mid)]
    mappers = pl.scalar_bars._scalar_bar_mappers['data']
    assert [m.scalar_range for m in mappers] == [(0.0, 1.0), (0.0, 1.0)]
    assert not any(m._use_default_scalar_range for m in mappers)
    actors.append(pl.add_mesh(high, scalars='data'))
    assert [m.scalar_range for m in mappers] == [(-1.0, 5.0)] * 3
    assert [m.lookup_table.scalar_range for m in mappers] == [(-1.0, 5.0)] * 3
    assert list(pl.scalar_bars._scalar_bar_ranges['data']) == [-1.0, 5.0]
    # A mesh inside the shared range leaves every mapper on it
    actors.append(pl.add_mesh(mid.copy(), scalars='data'))
    assert [m.scalar_range for m in mappers] == [(-1.0, 5.0)] * 4
    assert [m.lookup_table.scalar_range for m in mappers] == [(-1.0, 5.0)] * 4
    assert list(pl.scalar_bars._scalar_bar_ranges['data']) == [-1.0, 5.0]
    # A manual range update is undone by the next add, as before
    pl.update_scalar_bar_range([0, 10], name='data')
    assert [m.scalar_range for m in mappers] == [(0.0, 10.0)] * 4
    actors.append(pl.add_mesh(mid.copy(), scalars='data'))
    assert [m.scalar_range for m in mappers] == [(-1.0, 5.0)] * 5
    pl.update_scalar_bar_range([0, 10])
    actors.append(pl.add_mesh(mid.copy(), scalars='data'))
    assert [m.scalar_range for m in mappers] == [(-1.0, 5.0)] * 6
    pl.close()


def test_update_scalar_bar_range_without_a_bar(sphere):
    sphere['data'] = sphere.points[:, 2]
    pl = pv.Plotter()
    pl.add_mesh(sphere, scalars='data', show_scalar_bar=False)
    pl.update_scalar_bar_range([-1, 1])
    assert pl.mapper.scalar_range == (-1.0, 1.0)
    pl.close()


def test_remove_actor_removes_mapper_from_every_scalar_bar(sphere):
    """Test that removing an actor clears its mapper from all of its scalar bars."""
    sphere[KEY] = sphere.points[:, 2]
    pl = pv.Plotter()
    actor = pl.add_mesh(sphere, scalars=KEY)
    pl.add_scalar_bar('Second')
    mappers = pl.scalar_bars._scalar_bar_mappers
    assert [title for title, m in mappers.items() if actor.mapper in m] == [KEY, 'Second']

    pl.remove_actor(actor)
    assert [title for title, m in mappers.items() if actor.mapper in m] == []
    assert len(pl.scalar_bars) == 0
    pl.close()


def _wide_bar(pl, sphere, *, vertical=False, title=WIDE_KEY, **kwargs):
    """Add one scalar bar whose tick labels are wider than the bar has room for."""
    sphere[WIDE_KEY] = np.linspace(*WIDE_RANGE, sphere.n_points)
    return pl.add_scalar_bar(
        title,
        vertical=vertical,
        title_font_size=WIDE_FONT,
        label_font_size=WIDE_FONT,
        mapper=pv.DataSetMapper(sphere),
        **kwargs,
    )


def _drawn_ramp(pl, bar):
    """Return the pixels a render inks a horizontal bar's ramp across, below its blue text."""
    image = pl.screenshot(return_img=True)
    bottom = bar.GetPositionCoordinate().GetComputedViewportValue(pl.renderer)[1]
    height = _box_pixels(bar, pl.renderer)[1]
    box = image[image.shape[0] - int(bottom + height) : image.shape[0] - int(bottom)]
    # The ramp is the rows past the last row of text and the edge blurred under it
    ramp = box[np.flatnonzero(_blue_text(box).any(axis=1)).max() + 2 :]
    columns = np.flatnonzero(_colored(ramp).any(axis=0))
    return columns.max() - columns.min() + 1


def _label_gaps(pl, bar, *, font_size=None):
    """Return the pixels a horizontal bar leaves between each pair of its tick labels.

    The labels are laid out along the ramp as drawn, less a text pad, and measured as
    the whole pixels they cover, at the size they are drawn at or the one given.
    """
    viewport = pl.renderer
    label_text = bar.GetLabelTextProperty()
    font_size = label_text.GetFontSize() if font_size is None else font_size
    room = _drawn_ramp(pl, bar) - bar.GetTextPad()
    edges = []
    for anchor, text in sorted(_label_ticks(bar)):
        size = math.ceil(_text_size(viewport, label_text, text, font_size=font_size)[0]) + 1
        edges.append((room * anchor - size / 2, room * anchor + size / 2))
    return [after[0] - before[1] for before, after in itertools.pairwise(edges)]


def _text_at_the_viewport_edge(pl):
    """Return whether any blue text is drawn against an edge of a render."""
    text = _blue_text(pl.screenshot(return_img=True))
    assert text.any()
    edges = (text[0], text[-1], text[:, 0], text[:, -1])
    return any(bool(edge.any()) for edge in edges)


def _label_runs(pl, bar):
    """Return how many separated bands of text a vertical bar draws beside its ramp."""
    image = pl.screenshot(return_img=True)
    text = _blue_text(image)
    colored = _colored(image) & ~text
    left = bar.GetPositionCoordinate().GetComputedViewportValue(pl.renderer)[0]
    width = _box_pixels(bar, pl.renderer)[0]
    ramp = int(left + bar.GetBarRatio() * width)
    # The title is drawn above the ramp, the swatch annotations on its other side and an
    # outline around the box, so the text beside the rows the ramp is inked on is the
    # tick labels
    rows = colored[:, int(left) + 1 : ramp].any(axis=1)
    inked = text[rows, ramp : int(left + width) - 1].any(axis=1)
    return int(np.sum(inked[1:] & ~inked[:-1])) + int(inked[0])


@pytest.mark.parametrize(
    'window', [LARGE_WINDOW, HALF_WINDOW, THIRD_WINDOW], ids=['large', 'half', 'third']
)
def test_fit_fonts_holds_the_labels_apart(sphere, window):
    # Unconstrained labels are drawn at the size they ask for, so the size has to be one
    # that leaves each of them room on the bar, and the largest one that does
    pl = pv.Plotter(window_size=window)
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere, color='blue')
    pl.screenshot(return_img=True)

    font = bar.GetLabelTextProperty().GetFontSize()
    pad = bar.GetTextPad()
    assert font <= WIDE_FONT
    assert min(_label_gaps(pl, bar)) >= pad
    # One size larger and the labels would close up
    assert font == WIDE_FONT or min(_label_gaps(pl, bar, font_size=font + 1)) < pad


def test_fit_fonts_leaves_a_bar_with_room_alone(sphere):
    # A bar wide enough for its labels keeps every size it asked for
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=LARGE_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        KEY, n_labels=3, title_font_size=WIDE_FONT, label_font_size=WIDE_FONT, fmt='%.0f'
    )
    pl.screenshot(return_img=True)

    assert bar.GetTitleTextProperty().GetFontSize() == WIDE_FONT
    assert bar.GetLabelTextProperty().GetFontSize() == WIDE_FONT


@pytest.mark.parametrize('vertical', [True, False], ids=['vertical', 'horizontal'])
def test_fit_fonts_follow_the_viewport(sphere, vertical: bool):
    # The labels are measured in pixels while the bar is a fraction of the viewport, so
    # the fit only holds for the viewport it was measured against
    pl = pv.Plotter(window_size=LARGE_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere, vertical=vertical, n_labels=CROWDED_LABELS)
    pl.screenshot(return_img=True)
    large = bar.GetLabelTextProperty().GetFontSize()

    pl.window_size = SMALL_WINDOW
    pl.render()
    small = bar.GetLabelTextProperty().GetFontSize()
    assert small < large

    pl.window_size = LARGE_WINDOW
    pl.render()
    assert bar.GetLabelTextProperty().GetFontSize() == large

    # A refit lands where the first fit did
    pl.window_size = SMALL_WINDOW
    pl.render()
    assert bar.GetLabelTextProperty().GetFontSize() == small


def test_fit_fonts_take_a_size_set_by_hand(sphere):
    # A size set after the bar was added is the size the fit measures against from there,
    # so a refit lands on it rather than on the size the bar was given
    pl = pv.Plotter(window_size=LARGE_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere)
    pl.screenshot(return_img=True)
    assert bar.GetLabelTextProperty().GetFontSize() == WIDE_FONT

    by_hand = 6
    bar.GetLabelTextProperty().SetFontSize(by_hand)
    pl.window_size = SMALL_WINDOW
    pl.render()
    assert bar.GetLabelTextProperty().GetFontSize() == by_hand

    pl.window_size = LARGE_WINDOW
    pl.render()
    assert bar.GetLabelTextProperty().GetFontSize() == by_hand


def test_fit_fonts_leave_the_title_alone(sphere):
    # Nothing is drawn around the title, so it keeps its size however far it runs past
    # the bar it is centered on
    pl = pv.Plotter(window_size=SMALL_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere, title='Pressure at the inlet of the manifold (Pa)')
    pl.screenshot(return_img=True)

    title_text = bar.GetTitleTextProperty()
    assert title_text.GetFontSize() == WIDE_FONT
    assert (
        _text_size(pl.renderer, title_text, bar.GetTitle(), font_size=WIDE_FONT)[0]
        > (SMALL_WINDOW[0])
    )


def test_fit_fonts_leave_a_constrained_bar_to_vtk(sphere):
    # VTK sizes the text of a bar that draws a box around it, so the fit stays out of it
    pl = pv.Plotter(window_size=SMALL_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    boxed = _wide_bar(pl, sphere, outline=True)
    pl.screenshot(return_img=True)

    assert not boxed.GetUnconstrainedFontSize()
    assert boxed.GetLabelTextProperty().GetFontSize() == WIDE_FONT


@pytest.mark.parametrize(
    'window', [LARGE_WINDOW, HALF_WINDOW, SMALL_WINDOW], ids=['large', 'half', 'small']
)
def test_fit_fonts_hold_vertical_labels_apart(sphere, window):
    # A vertical bar stacks its labels along the ramp, so each has the one below it to
    # stay clear of
    pl = pv.Plotter(window_size=window)
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere, vertical=True, title='', n_labels=CROWDED_LABELS, color='blue')
    pl.screenshot(return_img=True)

    assert bar.GetLabelTextProperty().GetFontSize() <= WIDE_FONT
    assert _label_runs(pl, bar) == CROWDED_LABELS


def test_fit_fonts_measure_the_viewport_a_bar_is_drawn_in(sphere):
    # A bar's box is a fraction of its own viewport, so a subplot holds the text of a
    # bar that the whole window would leave hanging out of it
    pl = pv.Plotter(shape=(1, 2), window_size=LARGE_WINDOW, border=False)
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere, vertical=True, n_labels=CROWDED_LABELS, color='blue', outline=True)
    pl.screenshot(return_img=True)

    assert not _text_outside_the_box(pl, bar)


def test_fit_fonts_leave_a_label_over_the_viewport_edge_alone(sphere):
    # The labels on the ends of a bar drawn out to the sides of the viewport run past
    # them, and are left at their size for it as long as they clear each other
    # Scaled to labels wide enough to run past the viewport at this size
    sphere[KEY] = sphere.points[:, 2] * 255
    font_size = 40
    # The bar stops a tenth of the viewport short of either side
    margin = 0.1

    pl = pv.Plotter(window_size=SMALL_WINDOW)
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        KEY,
        n_labels=3,
        width=1 - 2 * margin,
        position_x=margin,
        label_font_size=font_size,
        title_font_size=font_size,
        color='blue',
    )
    pl.screenshot(return_img=True)

    assert bar.GetLabelTextProperty().GetFontSize() == font_size
    assert min(_label_gaps(pl, bar)) > 0
    assert _text_at_the_viewport_edge(pl)


def test_fit_fonts_keep_the_size_where_no_size_clears_the_labels(sphere):
    # Two ticks all but on the same spot overlap at any size, so shrinking would not help
    low, high = WIDE_RANGE
    pl = pv.Plotter(window_size=SMALL_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere, tick_locations=[low, low + 1.0, high])
    pl.screenshot(return_img=True)

    assert bar.GetLabelTextProperty().GetFontSize() == WIDE_FONT


def test_fit_fonts_skip_a_tick_drawn_on_another(sphere):
    # A tick given twice is drawn once, so it is not a label the others have to clear
    ticks = np.linspace(*WIDE_RANGE, 5).tolist()

    def fitted(tick_locations):
        pl = pv.Plotter(window_size=SMALL_WINDOW)
        pl.add_mesh(sphere, show_scalar_bar=False)
        bar = _wide_bar(pl, sphere, tick_locations=tick_locations)
        pl.screenshot(return_img=True)
        return bar.GetLabelTextProperty().GetFontSize()

    assert fitted(ticks) < WIDE_FONT
    assert fitted([*ticks, ticks[-1]]) == fitted(ticks)


def test_fit_fonts_fit_a_boxed_bar_to_the_box_it_grows(sphere):
    # A box grows around its text and seats the title in it, so the labels are fitted to
    # the room the grown box leaves them, on the first fit and again on a refit
    pl = pv.Plotter(window_size=SMALL_WINDOW)
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(
        pl,
        sphere,
        vertical=True,
        n_labels=CROWDED_LABELS,
        color='blue',
        outline=True,
        unconstrained_font_size=True,
    )

    def largest_that_fits():
        return _fitted_label_font(
            bar, vertical=True, title=bar.GetTitle(), viewport=pl.renderer, start=WIDE_FONT
        )

    pl.screenshot(return_img=True)
    first = bar.GetLabelTextProperty().GetFontSize()
    assert first < WIDE_FONT
    assert first == largest_that_fits()
    assert _label_runs(pl, bar) == CROWDED_LABELS

    pl.window_size = LARGE_WINDOW
    pl.render()
    pl.window_size = SMALL_WINDOW
    pl.render()
    assert bar.GetLabelTextProperty().GetFontSize() == first == largest_that_fits()


def test_fit_fonts_follow_a_title_set_by_hand(sphere):
    # A vertical bar gives up the end its title is drawn across, so a taller title set
    # after the fit leaves the labels less room and they are fitted again
    pl = pv.Plotter(window_size=SMALL_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere, vertical=True, n_labels=CROWDED_LABELS)
    pl.screenshot(return_img=True)
    before = bar.GetLabelTextProperty().GetFontSize()

    bar.GetTitleTextProperty().SetFontSize(3 * WIDE_FONT)
    pl.render()

    assert bar.GetLabelTextProperty().GetFontSize() < before


@pytest.mark.parametrize(
    'layout',
    [
        {'vertical': False},
        {'vertical': True},
        {'vertical': True, 'outline': True, 'unconstrained_font_size': True},
    ],
    ids=['horizontal', 'vertical', 'boxed'],
)
def test_fit_fonts_pad_the_title_by_the_labels_as_drawn(sphere, layout):
    # The padding is a share of the label size, so it follows the labels as they shrink
    title_pad = 0.5
    pl = pv.Plotter(window_size=SMALL_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _wide_bar(pl, sphere, n_labels=CROWDED_LABELS, title_pad=title_pad, **layout)
    pl.screenshot(return_img=True)

    label_font = bar.GetLabelTextProperty().GetFontSize()
    assert label_font < WIDE_FONT
    pad = round(title_pad * label_font)
    if layout.get('outline'):
        assert bar.GetVerticalTitleSeparation() == pad
    else:
        assert bar.GetTitleTextProperty().GetLineOffset() == -pad


def test_fit_fonts_skip_the_ticks_a_flat_range_hides(sphere):
    # VTK draws only the custom tick equal to the range when the range has no width, so
    # the ones it hides are not labels the fit has to find room for
    flat = 5.0
    sphere[WIDE_KEY] = np.full(sphere.n_points, flat)

    pl = pv.Plotter(window_size=LARGE_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        WIDE_KEY,
        tick_locations=[flat - 1, flat, flat + 1],
        label_font_size=WIDE_FONT,
        title_font_size=WIDE_FONT,
        mapper=pv.DataSetMapper(sphere),
    )
    pl.screenshot(return_img=True)

    # The tick at the value sits mid-ramp, and the others have no place on it
    assert [anchor for anchor, _ in _label_ticks(bar)] == [-1.0, 0.5, -1.0]
    assert bar.GetLabelTextProperty().GetFontSize() == WIDE_FONT


def test_label_ticks_place_custom_values_on_a_log_ramp(sphere):
    # A log ramp places a custom tick by its logarithm, and has nowhere to put one that
    # is not above zero
    sphere[KEY] = np.linspace(0.01, 100.0, sphere.n_points)

    pl = pv.Plotter()
    pl.add_mesh(sphere, log_scale=True, show_scalar_bar=False)
    bar = pl.add_scalar_bar(KEY, tick_locations=[0.0, 0.1, 10.0], fmt='%.2f')

    # The range spans four decades, and a tick sits as many quarters along as it is decades up
    assert _label_ticks(bar) == [(-1.0, '0.00'), (0.25, '0.10'), (0.75, '10.00')]


@pytest.mark.parametrize('indexed', [True, False], ids=['indexed', 'no_labels'])
def test_fit_fonts_leave_a_bar_with_no_tick_labels_alone(sphere, indexed: bool):
    # A bar that draws no tick labels has none to find room for
    clim = (0.0, 10.0)
    sphere[KEY] = np.linspace(*clim, sphere.n_points)
    table = pv.LookupTable(cmap='viridis')
    table.scalar_range = clim
    table.SetIndexedLookup(indexed)

    pl = pv.Plotter(window_size=SMALL_WINDOW)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        KEY,
        lookup_table=table,
        # An indexed lookup draws none of the five labels it is given; none turns them off
        n_labels=5 if indexed else 0,
        label_font_size=WIDE_FONT,
        title_font_size=WIDE_FONT,
    )
    pl.screenshot(return_img=True)

    assert bar.GetLabelTextProperty().GetFontSize() == WIDE_FONT


def test_fit_fonts_clear_the_swatches_a_vertical_bar_draws(sphere):
    # Each swatch takes its own end of the ramp, leaving the labels a shorter run

    def fit(**kwargs):
        pl = pv.Plotter(window_size=SMALL_WINDOW)
        pl.background_color = 'white'
        pl.add_mesh(sphere, show_scalar_bar=False)
        bar = _wide_bar(
            pl, sphere, vertical=True, title='', n_labels=CROWDED_LABELS, color='blue', **kwargs
        )
        return pl, bar

    plain, without = fit()
    bare = _label_runs(plain, without)

    pl, bar = fit(nan_annotation=True, below_label='low', above_label='high')

    assert bar.GetDrawNanAnnotation()
    assert bar.GetDrawBelowRangeSwatch()
    assert bar.GetDrawAboveRangeSwatch()
    assert bar.GetLabelTextProperty().GetFontSize() < without.GetLabelTextProperty().GetFontSize()
    assert _label_runs(pl, bar) == bare == CROWDED_LABELS
