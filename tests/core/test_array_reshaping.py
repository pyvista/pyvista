from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core._array_reshaping import ReshapedArrays

DIMENSIONS = (4, 3, 2)


def _image():
    return pv.ImageData(dimensions=DIMENSIONS)


def _rectilinear():
    return pv.RectilinearGrid(*(np.arange(n, dtype=float) for n in DIMENSIONS))


def _structured():
    return pv.StructuredGrid(
        *np.meshgrid(*(np.arange(n, dtype=float) for n in DIMENSIONS), indexing='ij')
    )


def _explicit():
    return _structured().cast_to_explicit_structured_grid()


GRIDS = [_image, _rectilinear, _structured, _explicit]
GRID_IDS = ['image', 'rectilinear', 'structured', 'explicit']


def _attributes(grid, association):
    return grid.point_data if association == 'point' else grid.cell_data


def _xyz_shape(association):
    return DIMENSIONS if association == 'point' else tuple(n - 1 for n in DIMENSIONS)


@pytest.mark.parametrize('make_grid', GRIDS, ids=GRID_IDS)
@pytest.mark.parametrize('association', ['point', 'cell'])
@pytest.mark.parametrize('n_components', [None, 3])
@pytest.mark.parametrize(
    ('preset', 'to_layout'),
    [
        ('grid', lambda a: a),
        ('zyx', lambda a: np.moveaxis(a, (0, 1, 2), (2, 1, 0))),
        ('channels_first', lambda a: np.moveaxis(a, -1, 0) if a.ndim == 4 else a),
    ],
)
def test_grid_presets_round_trip(make_grid, association, n_components, preset, to_layout):
    grid = make_grid()
    attributes = _attributes(grid, association)
    shape = _xyz_shape(association) + (() if n_components is None else (n_components,))
    xyz = np.arange(np.prod(shape)).reshape(shape)
    shaped = to_layout(xyz)

    attributes.reshaped.using(preset)['data'] = shaped

    flat = xyz.reshape((-1, *shape[3:]), order='F')
    assert np.array_equal(attributes['data'], flat)
    out = attributes.reshaped.using(preset)['data']
    assert out.shape == shaped.shape
    assert np.array_equal(out, shaped)
    assert np.shares_memory(out, attributes['data'])


def test_default_matches_manual_fortran_ravel():
    # layout from https://github.com/pyvista/pyvista/issues/4862
    dimensions = (1, 5, 3)
    grid = pv.ImageData(dimensions=dimensions)
    values = np.zeros(dimensions)
    values[:] = np.tile(np.arange(1, 6)[:, None], dimensions[2])
    grid.point_data.reshaped['V'] = values
    assert np.array_equal(grid['V'], values.ravel(order='F'))
    assert np.array_equal(grid.point_data.reshaped['V'], values)


def test_matches_pv_wrap_and_structured_grid_coordinates():
    values = np.random.default_rng(0).random(DIMENSIONS)
    wrapped = pv.wrap(values)
    assert np.array_equal(wrapped.point_data.reshaped['values'], values)

    grid = _structured()
    grid.point_data['x'] = grid.points[:, 0]
    assert np.array_equal(grid.point_data.reshaped['x'], grid.x)


def test_in_place_edit_reaches_dataset():
    grid = _image()
    grid.point_data['data'] = np.zeros(grid.n_points)
    grid.point_data.reshaped['data'][1, 0, 0] = 5
    assert grid.point_data['data'][1] == 5


@pytest.mark.parametrize(
    ('dimensions', 'association', 'expected'),
    [
        ((4, 3, 1), 'point', (4, 3, 1)),
        ((4, 3, 1), 'cell', (3, 2, 1)),
        ((4, 1, 1), 'cell', (3, 1, 1)),
    ],
)
def test_singleton_dimensions(dimensions, association, expected):
    grid = pv.ImageData(dimensions=dimensions)
    attributes = _attributes(grid, association)
    attributes['data'] = np.arange(attributes.valid_array_len)
    assert attributes.reshaped['data'].shape == expected

    squeezed = tuple(n for n in expected if n != 1)
    reshaper = pv.GridReshaper(squeeze=True)
    assert attributes.reshaped.using(reshaper)['data'].shape == squeezed
    attributes.reshaped.using(reshaper)['copy'] = attributes.reshaped.using(reshaper)['data']
    assert np.array_equal(attributes['copy'], attributes['data'])


@pytest.mark.parametrize('n_frames', [1, 3])
def test_image_preset_round_trip(n_frames):
    height, width = 5, 7
    shape = (height, width, 3) if n_frames == 1 else (n_frames, height, width, 3)
    pixels = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    grid = pv.ImageData(dimensions=(width, height, n_frames))
    grid.point_data.reshaped.using('image')['rgb'] = pixels
    assert np.array_equal(grid.point_data.reshaped.using('image')['rgb'], pixels)
    assert np.shares_memory(grid.point_data.reshaped.using('image')['rgb'], grid['rgb'])
    # first row of the image is the top row, at the largest y
    top_row = grid['rgb'].reshape(width, height, n_frames, 3, order='F')[:, -1, 0]
    first_image = pixels if n_frames == 1 else pixels[0]
    assert np.array_equal(top_row, first_image[0])


def test_image_preset_matches_png_reader_and_texture():
    from PIL import Image

    image = pv.read(examples.logofile)
    pixels = np.asarray(Image.open(examples.logofile))
    assert np.array_equal(image.point_data.reshaped.using('image')['PNGImage'], pixels)

    texture = pv.Texture(pixels)
    assert np.array_equal(texture.to_image().point_data.reshaped.using('image')['Image'], pixels)


def test_tensor_preset(sphere):
    tensors = np.random.default_rng(0).random((sphere.n_points, 3, 3))
    sphere.point_data.reshaped.using('tensor')['tensors'] = tensors
    assert sphere.point_data['tensors'].shape == (sphere.n_points, 9)
    assert np.array_equal(sphere.point_data.reshaped.using('tensor')['tensors'], tensors)

    # same layout as setting the 3D array directly
    sphere.point_data['direct'] = tensors
    assert np.array_equal(sphere.point_data['direct'], sphere.point_data['tensors'])

    sphere.point_data.reshaped.using('tensor')['scalars'] = np.arange(sphere.n_points)
    assert sphere.point_data.reshaped.using('tensor')['scalars'].shape == (sphere.n_points,)


@pytest.mark.parametrize('preset', ['grid', 'zyx'])
def test_squeezed_input_accepted(preset):
    grid = pv.ImageData(dimensions=(4, 3, 1))
    xy = np.arange(12).reshape(4, 3)
    pixels = xy.T if preset == 'zyx' else xy
    grid.point_data.reshaped.using(preset)['data'] = pixels
    assert np.array_equal(grid.point_data['data'], xy.ravel(order='F'))
    squeezed = pv.GridReshaper(axes='zyx' if preset == 'zyx' else 'xyz', squeeze=True)
    assert np.array_equal(grid.point_data.reshaped.using(squeezed)['data'], pixels)


def test_flat_input_passes_through():
    grid = _image()
    grid.point_data.reshaped['flat'] = np.arange(grid.n_points)
    grid.point_data.reshaped['vectors'] = grid.points
    grid.point_data.reshaped.using(pv.GridReshaper(squeeze=True))['squeezed'] = np.arange(
        grid.n_points
    )
    assert grid.point_data['squeezed'].shape == (grid.n_points,)
    assert grid.point_data['flat'].shape == (grid.n_points,)
    assert np.array_equal(grid.point_data['vectors'], grid.points)


def test_wrong_shape_raises():
    grid = _image()
    match = re.escape('Array of shape (2, 2) does not fit')
    with pytest.raises(ValueError, match=match):
        grid.point_data.reshaped['data'] = np.zeros((2, 2))
    with pytest.raises(ValueError, match='does not fit'):
        grid.point_data.reshaped['data'] = np.zeros((*DIMENSIONS, 3, 3))
    with pytest.raises(ValueError, match='does not fit'):
        grid.point_data.reshaped['data'] = np.zeros((5, 3, 2))


def test_grid_preset_on_non_grid_raises(sphere):
    sphere.point_data['data'] = np.arange(sphere.n_points)
    match = 'reshapes arrays of structured grids only, not arrays of PolyData'
    with pytest.raises(TypeError, match=match):
        sphere.point_data.reshaped['data']
    with pytest.raises(TypeError, match=match):
        sphere.point_data.reshaped['data'] = np.arange(sphere.n_points)


def test_grid_preset_on_field_data_raises():
    grid = _image()
    grid.field_data['data'] = np.arange(3)
    with pytest.raises(TypeError, match='point and cell arrays only, not NONE arrays'):
        grid.field_data.reshaped['data']


def test_custom_reshaper():
    calls = []

    def reshape(array, *, dataset, association):
        calls.append((type(dataset), association))
        return array.reshape(-1, 1)

    reshaper = pv.ArrayReshaper(reshape=reshape, flatten=lambda array, **_: array.ravel())
    grid = _image()
    grid.point_data.reshaped.using(reshaper)['data'] = np.arange(grid.n_points)[:, None]
    assert grid.point_data['data'].shape == (grid.n_points,)
    assert grid.point_data.reshaped.using(reshaper)['data'].shape == (grid.n_points, 1)
    assert calls == [(pv.ImageData, pv.FieldAssociation.POINT)]


def test_reshaper_follows_global_config():
    grid = _image()
    grid.point_data['data'] = np.arange(grid.n_points)
    view = grid.point_data.reshaped
    pv.global_config.array_reshaping = 'zyx'
    assert view['data'].shape == DIMENSIONS[::-1]
    assert view.reshaper.name == 'zyx'
    assert view.using('grid')['data'].shape == DIMENSIONS


def test_view_mapping_interface():
    grid = _image()
    view = grid.point_data.reshaped
    assert isinstance(view, ReshapedArrays)
    assert len(view) == 0
    assert 'Contains arrays : None' in repr(view)
    view['a'] = np.zeros(DIMENSIONS)
    view['b'] = np.zeros((*DIMENSIONS, 3))
    assert list(view) == ['a', 'b']
    assert 'a' in view
    assert 1 not in view
    assert view._ipython_key_completions_() == ['a', 'b']
    assert grid.point_data.active_scalars_name == 'a'
    text = repr(view)
    assert "name='grid'" in text
    assert '(4, 3, 2, 3)' in text
    del view['a']
    assert list(grid.point_data) == ['b']


def test_view_repr_with_unfit_arrays(sphere):
    sphere.point_data['data'] = np.arange(sphere.n_points)
    assert 'TypeError' in repr(sphere.point_data.reshaped)


@pytest.mark.parametrize(
    ('value', 'error', 'match'),
    [
        ('nope', ValueError, "Unknown array reshaping preset 'nope'"),
        (1, TypeError, 'must be a preset name or an ArrayReshaper, got int'),
    ],
)
def test_config_rejects_invalid(value, error, match):
    with pytest.raises(error, match=match):
        pv.global_config.array_reshaping = value
    assert pv.global_config.array_reshaping == 'grid'


def test_config_round_trip():
    reshaper = pv.GridReshaper(axes='zyx')
    pv.global_config.array_reshaping = reshaper
    assert pv.global_config.array_reshaping is reshaper
    restored = type(pv.global_config).from_dict(pv.global_config.to_dict())
    assert restored.array_reshaping is reshaper


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'axes': 'yxz'}, "`axes` must be 'xyz' or 'zyx'"),
        ({'components': 'middle'}, "`components` must be 'last' or 'first'"),
        ({'origin': 'top'}, "`origin` must be 'lower' or 'upper'"),
    ],
)
def test_grid_reshaper_rejects_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pv.GridReshaper(**kwargs)


def test_reprs():
    assert repr(pv.ArrayReshaper(reshape=np.ravel, flatten=np.ravel, name='x')) == (
        "ArrayReshaper(name='x')"
    )
    assert repr(pv.GridReshaper()) == (
        "GridReshaper(axes='xyz', components='last', squeeze=False, origin='lower')"
    )


@pytest.mark.parametrize(
    ('dimensions', 'expected'),
    [
        ((4, 3, 1), [[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]),
        ((4, 1, 3), [[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]]),
        ((1, 3, 4), [[9, 10, 11], [6, 7, 8], [3, 4, 5], [0, 1, 2]]),
    ],
    ids=['xy', 'xz', 'yz'],
)
def test_image_preset_flips_vertical_axis(dimensions, expected):
    grid = pv.ImageData(dimensions=dimensions)
    grid.point_data['data'] = np.arange(grid.n_points)
    assert np.array_equal(grid.point_data.reshaped.using('image')['data'], expected)
    grid.point_data.reshaped.using('image')['copy'] = np.array(expected)
    assert np.array_equal(grid.point_data['copy'], grid.point_data['data'])


def test_explicit_grid_with_cell_corners():
    # each cell has its own 8 corner points
    xyz = [np.repeat(np.arange(n, dtype=float), 2)[1:-1] for n in DIMENSIONS]
    corners = np.stack(
        [axis.ravel(order='F') for axis in np.meshgrid(*xyz, indexing='ij')], axis=1
    )
    grid = pv.ExplicitStructuredGrid(DIMENSIONS, corners)
    assert grid.n_points == 8 * grid.n_cells
    grid.point_data['data'] = np.arange(grid.n_points)
    with pytest.raises(TypeError, match='not one per grid index'):
        grid.point_data.reshaped['data']
    grid.cell_data['data'] = np.arange(grid.n_cells)
    assert grid.cell_data.reshaped['data'].shape == _xyz_shape('cell')


def test_components_first_with_every_axis_squeezed():
    grid = pv.ImageData(dimensions=(2, 2, 1))
    reshaper = pv.GridReshaper(components='first', squeeze=True)
    grid.cell_data.reshaped.using(reshaper)['rgb'] = np.array([1, 2, 3])
    assert np.array_equal(grid.cell_data['rgb'], [[1, 2, 3]])
    assert np.array_equal(grid.cell_data.reshaped.using(reshaper)['rgb'], [1, 2, 3])


def test_flat_input_preferred_over_squeezed_layout():
    grid = pv.ImageData(dimensions=(3, 1, 1))
    flat = np.arange(9.0).reshape(3, 3)
    grid.point_data.reshaped.using('channels_first')['data'] = flat
    assert np.array_equal(grid.point_data['data'], flat)


def test_grid_reshaper_is_read_only():
    reshaper = pv.GridReshaper()
    with pytest.raises(AttributeError):
        reshaper.axes = 'zyx'
