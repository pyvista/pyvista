from __future__ import annotations

import itertools

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk
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
    # A drawn box is sized without the padding, so the title would sit outside it
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

    assert pl.scalar_bar.GetTitleTextProperty().GetLineOffset() == 0


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


def test_stacked_vertical_bars_clear_an_uneven_neighbor(sphere):
    # A title claims half the gap on each side, so a long one clears its short neighbors
    sphere[KEY] = sphere.points[:, 2]
    window_size = [900, 400]
    titles = ['Short', 'A very much longer title', 'A bit long']

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(sphere, show_scalar_bar=False)
    bars = [
        pl.add_scalar_bar(title, vertical=True, title_font_size=18, mapper=pl.mapper)
        for title in titles
    ]

    dpi = pl.render_window.GetDPI()
    widths = [_title_width(b.GetTitleTextProperty(), b.GetTitle(), dpi) for b in bars]
    centers = [(b.GetPosition()[0] + b.GetWidth() / 2) * window_size[0] for b in bars]
    gap = 0.2 * pl.theme.colorbar_vertical.width * window_size[0]
    for (left, right), (left_width, right_width) in zip(
        itertools.pairwise(centers), itertools.pairwise(widths), strict=True
    ):
        assert left - right == pytest.approx(gap + (left_width + right_width) / 2)


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
