from __future__ import annotations

import itertools
import math
import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.core.errors import VTKVersionError
from pyvista.plotting.scalar_bars import _LEGIBLE_FONT_SIZE as LEGIBLE_FONT_SIZE
from pyvista.plotting.scalar_bars import _bar_title_height
from pyvista.plotting.scalar_bars import _box_pixels
from pyvista.plotting.scalar_bars import _fitting_font
from pyvista.plotting.scalar_bars import _label_size
from pyvista.plotting.scalar_bars import _label_texts
from pyvista.plotting.scalar_bars import _lifted_ramp
from pyvista.plotting.scalar_bars import _ramp_room
from pyvista.plotting.scalar_bars import _text_size
from pyvista.plotting.scalar_bars import _title_height
from pyvista.plotting.scalar_bars import _title_width

KEY = 'Data'


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
    pl.add_scalar_bar(KEY, vertical=vertical, title_font_size=font_size, title_pad=title_pad)

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
    pl.add_scalar_bar(KEY, vertical=vertical, title_font_size=font_size)

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
    font_size = 40
    bars = _wide_number_bars(
        pl, sphere, rotate_title=True, title_font_size=font_size, label_font_size=7
    )

    dpi = pl.render_window.GetDPI()
    pad = round(pl.theme.colorbar_vertical.title_pad * font_size)
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
    ramp = int(thickness - min(thickness / 8, text_pad))
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
    ramp = int(thickness - min(thickness / 8, text_pad))
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


def _ink_bands(pl, bar):
    """Return the heights of the runs of rows holding blue text across a bar's box, bottom up."""
    image = pl.screenshot(return_img=True)
    red, green, blue = (image[..., channel].astype(int) for channel in range(3))
    text = (blue > 200) & (red < 150) & (green < 150)
    left, _ = bar.GetPositionCoordinate().GetComputedViewportValue(pl.renderer)
    width = _box_pixels(bar, pl.renderer)[0]
    # The frame's sides run down every row, so they are left out of the columns read
    inked = text[::-1, left + 3 : left + width - 3].any(axis=1)
    bands = [len(list(run)) for holds_ink, run in itertools.groupby(inked) if holds_ink]
    # The frame's top and bottom are a row or two each
    return [band for band in bands if band > 2]


def _blue_ink(pl):
    """Return a mask of the pixels a render draws in blue."""
    image = pl.screenshot(return_img=True)
    red, green, blue = (image[..., channel].astype(int) for channel in range(3))
    return (blue > 200) & (red < 150) & (green < 150)


def _drawn_box(pl, bar):
    """Return the box drawn around a bar in pixels, as left, bottom, width and height."""
    left, bottom = bar.GetPositionCoordinate().GetComputedViewportValue(pl.renderer)
    return left, bottom, *_box_pixels(bar, pl.renderer)


def _flat_title(bar):
    """Return the property a bar's title is measured by when it is drawn flat."""
    probe = _vtk.vtkTextProperty()
    probe.ShallowCopy(bar.GetTitleTextProperty())
    probe.SetOrientation(0)
    probe.SetLineOffset(0)
    return probe


def _ramp_ends(pl, bar):
    """Return the rows a vertical bar's ramp runs between."""
    rect = [0, 0, 0, 0]
    bar.GetScalarBarRect(rect, pl.renderer)
    return rect[1], rect[1] + rect[3]


def _label_ink(pl, bar):
    """Return the runs of rows a vertical bar's labels ink, bottom up."""
    text = _blue_ink(pl)
    rect = [0, 0, 0, 0]
    bar.GetScalarBarRect(rect, pl.renderer)
    width = _label_size(bar, bar.GetLabelTextProperty(), pl.render_window.GetDPI())[0]
    # The labels are drawn from the far side of the ramp, and a turned title past them
    left = rect[0] + rect[2] + 2
    inked = text[::-1, left : left + int(width) + 2].any(axis=1)
    runs = []
    row = 0
    for holds_ink, run in itertools.groupby(inked):
        length = len(list(run))
        if holds_ink and length > 2:
            runs.append((row, row + length - 1))
        row += length
    return runs


def _label_centers(pl, bar):
    """Return the rows the ink of a vertical bar's top and bottom labels is centered on."""
    runs = _label_ink(pl, bar)
    return sum(runs[-1]) / 2, sum(runs[0]) / 2


def _text_outside_the_box(pl, bar):
    """Return whether any of a bar's blue text is drawn outside the box drawn around it."""
    text = _blue_ink(pl)
    left, bottom, width, height = _drawn_box(pl, bar)
    rows = text.shape[0]
    inside = text.copy()
    inside[rows - bottom - height - 2 : rows - bottom + 2, left - 2 : left + width + 2] = False
    assert text.any()
    return bool(inside.any())


def _registered(pl, bar):
    """Return whether a vertical bar's end labels are centered on the ends of its ramp.

    VTK centers them there itself, so this holds the fit to leaving them alone.
    """
    # The ramp is laid out by the render the ink is read from
    top, bottom = _label_centers(pl, bar)
    ramp_bottom, ramp_top = _ramp_ends(pl, bar)
    label_height = _label_size(bar, bar.GetLabelTextProperty(), pl.render_window.GetDPI())[1]
    return max(abs(top - ramp_top), abs(bottom - ramp_bottom)) <= label_height / 4


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
@pytest.mark.filterwarnings('ignore:The text of scalar bar')
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
    # Text asked to keep its size is left to it, box or no box
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(
        pl, sphere, vertical=False, box={'outline': True}, unconstrained_font_size=True
    )
    pl.screenshot(return_img=True)

    assert bar.GetUnconstrainedFontSize()
    pad = round(pl.theme.colorbar_horizontal.title_pad * 24)
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


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_stacked_turned_bars_clear_the_boxes_beside_them(sphere):
    # A box holds a turned title, so the bar beside it clears the box, not the title
    sphere[KEY] = sphere.points[:, 2]

    def positions(**box):
        pl = pv.Plotter(window_size=[1024, 768])
        pl.add_mesh(sphere, show_scalar_bar=False)
        bars = [
            pl.add_scalar_bar(
                title,
                vertical=True,
                rotate_title=True,
                title_font_size=24,
                label_font_size=24,
                n_labels=5,
                mapper=pv.DataSetMapper(sphere),
                **box,
            )
            for title in ('Alpha', 'Beta and gamma', 'Delta')
        ]
        pl.screenshot(return_img=True)
        places = [(bar.GetPosition()[0], bar.GetWidth()) for bar in bars]
        pl.close()
        return places

    boxed = positions(outline=True)
    bare = positions()
    # Each box stands a fifth of itself from the next, whatever the title inside it says
    for (place, width), (nearer, _) in zip(boxed[1:], boxed[:-1], strict=True):
        assert nearer - place == pytest.approx(width * 1.2, abs=0.01)
    # A box is wider than the bar inside it, so the bars stand further apart, not nearer
    assert boxed[0][0] - boxed[1][0] > bare[0][0] - bare[1][0]


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_stacked_bar_clears_a_turned_neighbor_by_the_height_of_its_title(sphere):
    # A title reaching into the gap is one line tall whatever it says
    sphere[KEY] = sphere.points[:, 2]
    turned = dict(
        vertical=True, rotate_title=True, title_font_size=24, label_font_size=24, n_labels=5
    )

    def beside(title):
        pl = pv.Plotter(window_size=[1024, 768])
        pl.add_mesh(sphere, show_scalar_bar=False)
        neighbor = pl.add_scalar_bar(
            title, outline=True, mapper=pv.DataSetMapper(sphere), **turned
        )
        pl.screenshot(return_img=True)
        # The box comes off between renders, so the fit is what the neighbor is measured by
        neighbor.SetDrawFrame(False)
        bar = pl.add_scalar_bar('Delta', mapper=pv.DataSetMapper(sphere), **turned)
        place = bar.GetPosition()[0]
        pl.close()
        return place

    assert beside('Beta and gamma and delta') == pytest.approx(beside('Beta'), abs=0.005)


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


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
def test_fit_box_seats_a_sized_vertical_title(sphere, box):
    # A vertical box kept at a size of its own holds its title inside itself rather than
    # across the top edge the layout lifts the title through
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(
        pl,
        sphere,
        vertical=True,
        box=box,
        fmt='%.1f',
        width=0.25,
        height=0.6,
        position_x=0.6,
        position_y=0.2,
        color='blue',
    )

    assert bar.GetWidth() == pytest.approx(0.25)
    assert bar.GetHeight() == pytest.approx(0.6)
    assert bar.GetTitleTextProperty().GetFontSize() == 24
    assert bar.GetTitleTextProperty().GetLineOffset() > 0
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
def test_fit_box_widens_a_vertical_bar_given_only_a_height(sphere, box):
    # A vertical title spans the width of its box, so only a width of its own holds the
    # text to it; a height leaves the box free to widen around the title at full size
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(pl, sphere, vertical=True, box=box, fmt='%.1f', height=0.6, color='blue')

    dpi = pl.render_window.GetDPI()
    assert bar.GetHeight() == pytest.approx(0.6)
    assert bar.GetWidth() > 0.08
    assert _box_pixels(bar, pl.renderer)[0] >= _title_width(
        bar.GetTitleTextProperty(), FIT_TITLE, dpi
    )
    assert bar.GetTitleTextProperty().GetFontSize() == 24
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
def test_fit_box_encloses_a_turned_title(sphere, box):
    # A turned title is drawn alongside the bar, past the tick labels, and the box holds
    # the ramp, the labels and the title in a row rather than leaving the title outside,
    # reaching past the ramp's ends for the labels centered on them
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(
        pl, sphere, vertical=True, box=box, rotate_title=True, fmt='%.1f', color='blue'
    )

    dpi = pl.render_window.GetDPI()
    title_text = bar.GetTitleTextProperty()
    # The title is turned by its text property and moved beside the labels rather than
    # to the far side of the bar, and the spaces that move it along the bar are carried
    # past the title rather than in it
    assert not bar.GetForceVerticalTitle()
    assert title_text.GetOrientation() == 90
    assert title_text.GetLineOffset() > 0
    assert bar.GetTitle() == FIT_TITLE
    ramp = bar.GetBarRatio() * _box_pixels(bar, pl.renderer)[0]
    labels = _label_size(bar, bar.GetLabelTextProperty(), dpi)[0]
    # The property measures the title turned and offset, so measure it flat and in place
    turned = _title_height(_flat_title(bar), FIT_TITLE, dpi)
    assert _box_pixels(bar, pl.renderer)[0] >= ramp + labels + turned
    # The box grows around the ramp by the half label reaching past its top
    label_height = _label_size(bar, bar.GetLabelTextProperty(), dpi)[1]
    asked = pl.theme.colorbar_vertical.height * 768
    assert _box_pixels(bar, pl.renderer)[1] >= asked + label_height / 2
    assert _registered(pl, bar)
    assert not _text_outside_the_box(pl, bar)
    # The box reaches no further past the top label than the pad and the line, and the
    # few pixels the ink of a digit stops short of its bounds by
    _, bottom, _, height = _drawn_box(pl, bar)
    top_ink = _label_ink(pl, bar)[-1][1]
    line_width = int(bar.GetFrameProperty().GetLineWidth())
    assert bottom + height - top_ink <= bar.GetTextPad() + line_width + 3


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
@pytest.mark.parametrize(
    ('asked', 'gives_way'),
    [((24, 24), 'both'), ((24, 12), 'title'), ((12, 24), 'labels')],
    ids=['same', 'larger-title', 'larger-labels'],
)
def test_fit_box_shares_a_given_width_between_a_turned_title_and_the_labels(
    sphere, asked, gives_way
):
    # A width of its own is the room the row past the ramp has, and the one that asked
    # for the larger size gives way first, both together once they match
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = pl.add_scalar_bar(
        FIT_TITLE,
        vertical=True,
        outline=True,
        rotate_title=True,
        fmt='%.1f',
        title_font_size=asked[0],
        label_font_size=asked[1],
        n_labels=5,
        mapper=pv.DataSetMapper(sphere),
        width=0.08,
        color='blue',
    )

    title = bar.GetTitleTextProperty().GetFontSize()
    labels = bar.GetLabelTextProperty().GetFontSize()
    assert bar.GetWidth() == pytest.approx(0.08)
    if gives_way == 'both':
        assert title == labels < 24
    elif gives_way == 'title':
        assert labels == 12
        assert 12 < title < 24
    else:
        assert title == 12
        assert 12 < labels < 24
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_keeps_a_turned_bar_the_height_it_was_given(sphere):
    # The box keeps the height it was given, and the ramp gives the labels their room
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(
        pl,
        sphere,
        vertical=True,
        box={'outline': True},
        rotate_title=True,
        fmt='%.1f',
        height=0.6,
        color='blue',
    )
    pl.screenshot(return_img=True)

    assert _box_pixels(bar, pl.renderer)[1] == pytest.approx(0.6 * 768, abs=1)
    # The ramp is held back from the top for the label centered on its end
    label_height = _label_size(bar, bar.GetLabelTextProperty(), pl.render_window.GetDPI())[1]
    _, bar_bottom = bar.GetPositionCoordinate().GetComputedViewportValue(pl.renderer)
    assert _ramp_ends(pl, bar)[1] <= bar_bottom + 0.6 * 768 - label_height / 2
    assert _registered(pl, bar)
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_lets_a_swatch_hold_a_turned_bar_label(sphere):
    # A swatch drawn above the ramp already holds the top label off the frame, so the
    # box has no half label to grow by.  VTK draws the swatch's annotation on the far
    # side of the ramp, outside any box, so the ink is not checked
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(
        pl,
        sphere,
        vertical=True,
        box={'outline': True},
        rotate_title=True,
        fmt='%.1f',
        above_label='over',
        color='blue',
    )
    pl.screenshot(return_img=True)

    asked = pl.theme.colorbar_vertical.height * 768
    assert _box_pixels(bar, pl.renderer)[1] <= asked + 4
    assert _registered(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_moves_a_turned_bar_box_with_it(sphere):
    # A bar placed after it is added takes its box along
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(
        pl,
        sphere,
        vertical=True,
        box={'outline': True},
        rotate_title=True,
        fmt='%.1f',
        color='blue',
    )
    pl.screenshot(return_img=True)
    bar.SetPosition(0.1, 0.2)
    pl.screenshot(return_img=True)

    left, bottom, width, _ = _drawn_box(pl, bar)
    # A vertical box is anchored at its right edge and grows away from it
    assert bottom == pytest.approx(0.2 * 768, abs=1)
    assert left + width == pytest.approx((0.1 + pl.theme.colorbar_vertical.width) * 1024, abs=1)
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
@pytest.mark.parametrize('gone', ['hidden', 'removed', 'taken out'])
def test_fit_box_takes_a_turned_bar_box_away_with_it(sphere, gone):
    # The box is drawn only while the bar is
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(
        pl,
        sphere,
        vertical=True,
        box={'outline': True},
        rotate_title=True,
        fmt='%.1f',
        color='blue',
    )
    assert _blue_ink(pl).any()

    if gone == 'hidden':
        bar.SetVisibility(False)
    elif gone == 'removed':
        pl.remove_scalar_bar(FIT_TITLE)
    else:
        pl.remove_actor(bar)

    assert not _blue_ink(pl).any()
    if gone == 'removed':
        # A bar taken off the plotter is given back the layout it was added with
        title_text = bar.GetTitleTextProperty()
        assert bar.GetForceVerticalTitle()
        assert title_text.GetOrientation() == 0
        assert title_text.GetLineOffset() == 0
        assert bar.GetVerticalTitleSeparation() >= 0
        assert bar.GetComponentTitle() == ''


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
@pytest.mark.parametrize('box', [{}, {'height': 0.45}], ids=['grows', 'kept'])
def test_fit_box_holds_a_long_turned_title(sphere, box):
    # A title longer than the ramp reaches past its ends too, so the box grows around
    # it, unless the box was given a height, which holds the title to it instead
    sphere[KEY] = sphere.points[:, 2]
    title = 'Elevation above the reference ellipsoid'

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = pl.add_scalar_bar(
        title,
        vertical=True,
        outline=True,
        rotate_title=True,
        fmt='%.1f',
        title_font_size=24,
        label_font_size=24,
        n_labels=5,
        mapper=pv.DataSetMapper(sphere),
        color='blue',
        **box,
    )
    pl.screenshot(return_img=True)

    dpi = pl.render_window.GetDPI()
    # The property measures the title turned, so the length it is grown around is flat
    length = _title_width(_flat_title(bar), title, dpi)
    if box:
        assert bar.GetTitleTextProperty().GetFontSize() < 24
        assert _drawn_box(pl, bar)[3] == pytest.approx(0.45 * 768, abs=1)
    else:
        assert bar.GetTitleTextProperty().GetFontSize() == 24
        assert _drawn_box(pl, bar)[3] > length
    assert _registered(pl, bar)
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
@pytest.mark.parametrize(
    'placed',
    [{'title_font_size': 40, 'label_font_size': 40}, {'position_x': 0.0}],
    ids=['upward', 'leftward'],
)
def test_fit_box_grows_a_turned_box_no_further_than_the_window(sphere, placed):
    # The box grows around a title longer than it, and the window is as far as it grows
    sphere[KEY] = sphere.points[:, 2]
    title = 'Elevation above the reference ellipsoid'

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    kwargs = dict(
        vertical=True,
        outline=True,
        rotate_title=True,
        fmt='%.1f',
        title_font_size=24,
        label_font_size=24,
        n_labels=5,
        color='blue',
    )
    bar = pl.add_scalar_bar(title, mapper=pv.DataSetMapper(sphere), **{**kwargs, **placed})
    pl.screenshot(return_img=True)

    left, bottom, width, height = _drawn_box(pl, bar)
    assert left >= 0
    assert left + width <= 1024
    assert bottom + height <= 768
    if 'position_x' not in placed:
        # The title is held to the box the window would not let grow around it
        assert bar.GetTitleTextProperty().GetFontSize() < 40


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_keeps_a_component_title_given_to_a_turned_bar(sphere):
    # The padding that places the title along the bar is not the bar's component title
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(
        pl, sphere, vertical=True, box={'outline': True}, rotate_title=True, fmt='%.1f'
    )
    pl.screenshot(return_img=True)
    bar.SetComponentTitle('m/s')
    pl.window_size = [800, 600]
    pl.screenshot(return_img=True)

    assert bar.GetComponentTitle().rstrip(' ') == 'm/s'
    assert bar.GetComponentTitle() != 'm/s'
    assert bar.GetTitle() == FIT_TITLE

    pl.remove_scalar_bar(FIT_TITLE)
    assert bar.GetComponentTitle() == 'm/s'


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_leaves_a_title_of_several_lines_to_the_bar(sphere):
    # Spaces shear the lines of a title apart, so the bar is left to turn it itself
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = pl.add_scalar_bar(
        'Surface\nElevation',
        vertical=True,
        outline=True,
        rotate_title=True,
        fmt='%.1f',
        title_font_size=24,
        label_font_size=24,
        n_labels=5,
        mapper=pv.DataSetMapper(sphere),
    )
    pl.screenshot(return_img=True)

    assert bar.GetForceVerticalTitle()
    assert bar.GetTitleTextProperty().GetOrientation() == 0
    assert not bar.GetComponentTitle()


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_lays_out_a_turned_bar_with_no_title(sphere):
    # There is no title to move along the bar, so the box holds the ramp and the labels
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = pl.add_scalar_bar(
        '',
        vertical=True,
        outline=True,
        rotate_title=True,
        fmt='%.1f',
        title_font_size=24,
        label_font_size=24,
        n_labels=5,
        mapper=pv.DataSetMapper(sphere),
        color='blue',
    )
    pl.screenshot(return_img=True)

    assert bar.GetTitle() == ''
    # Nothing is carried past the title, so the component title stays empty
    assert not bar.GetComponentTitle()
    assert _registered(pl, bar)
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_keeps_a_turned_bar_the_box_it_was_given(sphere):
    # Given both, the box keeps both and the text is held to what they leave
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(
        pl,
        sphere,
        vertical=True,
        box={'outline': True, 'width': 0.08, 'height': 0.45},
        rotate_title=True,
        fmt='%.1f',
        color='blue',
    )
    pl.screenshot(return_img=True)

    width, height = _box_pixels(bar, pl.renderer)
    assert width == pytest.approx(0.08 * 1024, abs=1)
    assert height == pytest.approx(0.45 * 768, abs=1)
    assert bar.GetTitleTextProperty().GetFontSize() < 24
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_holds_a_turned_bar_too_small_for_any_font(sphere):
    # The text stops at a size that can still be read, and says that it does not fit
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.add_mesh(sphere, show_scalar_bar=False)
    with pytest.warns(UserWarning, match='does not fit its box'):
        bar = _fitted_bar(
            pl,
            sphere,
            vertical=True,
            box={'outline': True, 'width': 0.02, 'height': 0.12},
            rotate_title=True,
            fmt='%.1f',
        )
    pl.screenshot(return_img=True)

    assert bar.GetTitleTextProperty().GetFontSize() == LEGIBLE_FONT_SIZE
    assert bar.GetLabelTextProperty().GetFontSize() == LEGIBLE_FONT_SIZE


def test_fit_box_warns_once_that_a_title_does_not_fit_its_box(sphere):
    # A box too narrow for the title keeps it legible and overflowing, and says so once
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.add_mesh(sphere, show_scalar_bar=False)
    warning = re.escape(f'scalar bar {FIT_TITLE!r} does not fit its box')
    box = {'outline': True, 'width': 0.03, 'height': 0.4}
    with pytest.warns(UserWarning, match=warning):
        bar = _fitted_bar(pl, sphere, vertical=True, box=box)
    pl.screenshot(return_img=True)

    assert bar.GetTitleTextProperty().GetFontSize() == LEGIBLE_FONT_SIZE
    # The refit a resized window runs says nothing more about a box already reported
    pl.window_size = [900, 700]
    pl.screenshot(return_img=True)
    pl.close()


def test_fit_box_sizes_a_box_with_no_tick_labels(sphere):
    # There are no labels to shrink, so the width it was given holds the title alone
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = pl.add_scalar_bar(
        FIT_TITLE,
        vertical=True,
        outline=True,
        width=0.1,
        tick_locations=[],
        title_font_size=24,
        label_font_size=24,
        mapper=pv.DataSetMapper(sphere),
        color='blue',
    )
    pl.screenshot(return_img=True)

    assert _label_texts(bar) == []
    assert _box_pixels(bar, pl.renderer)[0] == pytest.approx(0.1 * 1024, abs=1)
    # The title is held to the width, which it no longer shares with any label
    assert bar.GetTitleTextProperty().GetFontSize() < 24
    assert not _text_outside_the_box(pl, bar)


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_leaves_a_turned_title_alone_without_a_box(sphere):
    # With nothing drawn around it the title keeps the far side of the bar to itself
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(pl, sphere, vertical=True, box={}, rotate_title=True, fmt='%.1f')
    pl.screenshot(return_img=True)

    assert bar.GetTitleTextProperty().GetLineOffset() < 0


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
def test_fit_box_frees_a_turned_title_with_its_box(sphere):
    # A box taken off a bar after it is added leaves the turned title the far side of
    # the bar, where a bar added without a box has it
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(
        pl, sphere, vertical=True, box={'outline': True}, rotate_title=True, fmt='%.1f'
    )
    pl.screenshot(return_img=True)
    bar.SetDrawFrame(False)
    pl.screenshot(return_img=True)

    bare = pv.Plotter(window_size=[1024, 768])
    bare.add_mesh(sphere, show_scalar_bar=False)
    twin = _fitted_bar(bare, sphere, vertical=True, box={}, rotate_title=True, fmt='%.1f')
    bare.screenshot(return_img=True)
    bare.close()

    offset = bar.GetTitleTextProperty().GetLineOffset()
    assert offset == twin.GetTitleTextProperty().GetLineOffset()
    assert offset < 0
    assert bar.GetWidth() == twin.GetWidth()
    # The bar turns the title itself again
    assert bar.GetForceVerticalTitle()
    assert bar.GetTitleTextProperty().GetOrientation() == 0
    assert bar.GetComponentTitle() == ''


def test_fit_box_shrinks_a_sized_vertical_title(sphere):
    # A title wider than the box it was given is shrunk to it, and the labels keep the
    # size they asked for
    sphere[KEY] = sphere.points[:, 2]
    title = 'Elevation above the reference ellipsoid, in metres'

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = pl.add_scalar_bar(
        title,
        vertical=True,
        outline=True,
        fmt='%.1f',
        width=0.25,
        height=0.6,
        position_x=0.6,
        position_y=0.2,
        title_font_size=24,
        label_font_size=20,
        n_labels=5,
        color='blue',
        mapper=pv.DataSetMapper(sphere),
    )

    assert bar.GetTitleTextProperty().GetFontSize() < 24
    assert bar.GetLabelTextProperty().GetFontSize() == 20
    assert not _text_outside_the_box(pl, bar)


def test_fit_box_shrinks_sized_vertical_labels(sphere):
    # Labels wider than the room left beside the ramp are shrunk to it, and the title
    # keeps the size it asked for
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1024, 768])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = pl.add_scalar_bar(
        'Depth',
        vertical=True,
        outline=True,
        fmt='%.6f',
        width=0.12,
        height=0.6,
        position_x=0.6,
        position_y=0.2,
        title_font_size=24,
        label_font_size=20,
        n_labels=5,
        color='blue',
        mapper=pv.DataSetMapper(sphere),
    )

    assert bar.GetLabelTextProperty().GetFontSize() < 20
    assert bar.GetTitleTextProperty().GetFontSize() == 24
    assert not _text_outside_the_box(pl, bar)


def test_fit_box_refits_a_sized_vertical_bar(sphere):
    # The box is a fraction of the window and the text is not, so the text is fitted to
    # the box again whenever the window changes, and gets its size back when there is room
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[1400, 700])
    pl.add_mesh(sphere, show_scalar_bar=False)
    bar = _fitted_bar(
        pl, sphere, vertical=True, box={'outline': True}, fmt='%.1f', width=0.15, height=0.6
    )
    pl.screenshot(return_img=True)
    assert bar.GetTitleTextProperty().GetFontSize() == 24

    pl.window_size = [500, 700]
    pl.screenshot(return_img=True)
    assert bar.GetTitleTextProperty().GetFontSize() < 24

    pl.window_size = [1400, 700]
    pl.screenshot(return_img=True)
    assert bar.GetTitleTextProperty().GetFontSize() == 24


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
def test_fit_box_keeps_a_font_set_on_the_actor(sphere, box):
    # A font size set after the bar is added is the one it asks for, so the next fit
    # measures the text at that size rather than putting the first one back
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter(window_size=[900, 700])
    pl.background_color = 'white'
    pl.add_mesh(sphere, show_scalar_bar=False, cmap='autumn')
    bar = _fitted_bar(pl, sphere, vertical=True, box=box, color='blue')
    pl.screenshot(return_img=True)
    bar.GetTitleTextProperty().SetFontSize(36)
    bar.GetLabelTextProperty().SetFontSize(12)
    pl.screenshot(return_img=True)

    assert bar.GetTitleTextProperty().GetFontSize() == 36
    assert bar.GetLabelTextProperty().GetFontSize() == 12
    assert not _text_outside_the_box(pl, bar)

    # A refit of the box, here for a new window, measures against those sizes too
    pl.window_size = [1000, 700]
    pl.screenshot(return_img=True)

    assert bar.GetTitleTextProperty().GetFontSize() == 36
    assert bar.GetLabelTextProperty().GetFontSize() == 12
    assert not _text_outside_the_box(pl, bar)


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
@pytest.mark.usefixtures('verify_image_cache')
def test_fit_box_height_only_vertical_render(sphere, box):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    _fitted_bar(pl, sphere, vertical=True, box=box, fmt='%.1f', height=0.6)
    pl.show()


@pytest.mark.needs_vtk_version(9, 4, 0, reason='ForceVerticalTitle was added in VTK 9.4.0')
@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
@pytest.mark.usefixtures('verify_image_cache')
def test_fit_box_turned_title_render(sphere, box):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    _fitted_bar(pl, sphere, vertical=True, box=box, rotate_title=True, fmt='%.1f')
    pl.show()


@pytest.mark.parametrize('box', BOXES, ids=BOX_IDS)
@pytest.mark.usefixtures('verify_image_cache')
def test_fit_box_sized_vertical_render(sphere, box):
    sphere[KEY] = sphere.points[:, 2]

    pl = pv.Plotter()
    pl.add_mesh(sphere, show_scalar_bar=False)
    _fitted_bar(pl, sphere, vertical=True, box=box, width=0.15, height=0.6, position_x=0.7)
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
