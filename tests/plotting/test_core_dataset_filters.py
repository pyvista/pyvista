"""Tests for core dataset filters that require plotting color utilities."""

from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting.colors import matplotlib_default_colors


@pytest.fixture
def labeled_image():
    image = pv.ImageData(dimensions=(2, 2, 2))
    image['labels'] = [0, 3, 3, 3, 3, 0, 2, 2]
    return image


@pytest.mark.parametrize('coloring_mode', ['index', 'cycle', None])
def test_color_labels(uniform, coloring_mode):
    default_cmap = pv.get_cmap_safe('glasbey_category10')
    original_scalars_name = uniform.active_scalars_name

    if coloring_mode == 'index':
        match = (
            "Index coloring mode cannot be used with scalars 'Spatial Point Data'. "
            'Scalars must be positive integers \n'
            'and the max value (729.0) must be less than the number of colors (256).'
        )
        with pytest.raises(ValueError, match=re.escape(match)):
            uniform.color_labels(coloring_mode=coloring_mode)

        uniform = uniform.pack_labels(output_scalars=original_scalars_name)

    colored_mesh = uniform.color_labels(coloring_mode=coloring_mode)
    assert colored_mesh is not uniform
    assert uniform.active_scalars_name == original_scalars_name
    colors_name = original_scalars_name + '_rgb'
    assert [0, 0, 0] not in np.unique(colored_mesh[colors_name], axis=0).tolist()

    label_ids = np.unique(uniform.active_scalars)
    for i, label_id in enumerate(label_ids):
        data_ids = np.where(uniform[original_scalars_name] == label_id)[0]
        expected_color_rgb = pv.Color(default_cmap.colors[i]).int_rgb
        for data_id in data_ids:
            actual_rgba = colored_mesh[colors_name][data_id]
            assert np.allclose(actual_rgba, expected_color_rgb)

    colored_mesh = uniform.color_labels(coloring_mode=coloring_mode, inplace=True)
    assert colored_mesh is uniform


VIRIDIS_RGB = [pv.Color(c).int_rgb for c in pv.get_cmap_safe('viridis').colors]
COLORS_DICT = {
    0: 'red',
    1: (0, 0, 0),
    2: 'blue',
    3: (1.0, 1.0, 1.0),
    4: 'orange',
    5: 'green',
}
COLORS_DICT_RGB = [pv.Color(c).int_rgb for c in COLORS_DICT.values()]
RED_RGB = pv.Color('red').int_rgb


@pytest.mark.parametrize(
    ('color_input', 'expected_rgb'),
    [
        ('viridis', VIRIDIS_RGB),
        (pv.get_cmap_safe('viridis'), VIRIDIS_RGB),
        (COLORS_DICT, COLORS_DICT_RGB),
        (COLORS_DICT_RGB, COLORS_DICT_RGB),
        ('red', [RED_RGB, RED_RGB, RED_RGB, RED_RGB]),
    ],
    ids=['cmap_str', 'cmap_instance', 'dict', 'sequence', 'named_color'],
)
def test_color_labels_inputs(labeled_image, color_input, expected_rgb):
    label_scalars = labeled_image.active_scalars
    colored = labeled_image.color_labels(color_input)
    color_scalars = colored.active_scalars
    for id_ in np.unique(label_scalars):
        assert np.allclose(color_scalars[label_scalars == id_], expected_rgb[id_])


@pytest.mark.parametrize('color_type', ['int_rgb', 'int_rgba', 'float_rgb', 'float_rgba'])
def test_color_labels_color_type_partial_dict(labeled_image, color_type):
    input_scalars_name = labeled_image.active_scalars_name
    colored = labeled_image.color_labels({0: RED_RGB}, color_type=color_type)
    color_scalars = colored.active_scalars
    color_scalars_name = colored.active_scalars_name
    unique = np.unique(color_scalars, axis=0)

    expected_color = getattr(pv.Color(RED_RGB), color_type)
    if 'float' in color_type:
        assert np.array_equal(expected_color, unique[0])
        assert np.array_equal([np.nan] * len(expected_color), unique[1], equal_nan=True)
        assert color_scalars.dtype == float
    else:
        assert np.array_equal(expected_color, unique[1])
        assert np.array_equal([0] * len(expected_color), unique[0])
        assert color_scalars.dtype == np.uint8
    if 'rgba' in color_type:
        assert color_scalars_name == input_scalars_name + '_rgba'
    else:
        assert color_scalars_name == input_scalars_name + '_rgb'


@pytest.mark.parametrize('color_type', ['float_rgb', 'float_rgba'])
def test_color_labels_color_type_cmap(color_type):
    labels = pv.ImageData(dimensions=(256, 1, 1))
    labels['256'] = range(256)
    colored = labels.color_labels('viridis', color_type=color_type)
    cmap_colors = pv.get_cmap_safe('viridis').colors
    for i, color in enumerate(colored.active_scalars):
        expected_color = cmap_colors[i]
        if 'rgba' in color_type:
            expected_color.append(1.0)
        assert np.array_equal(color, expected_color)


LABEL_DATA = [-1, -2, 1]


@pytest.mark.parametrize(
    ('negative_indexing', 'cmap_index'),
    [(True, LABEL_DATA), (False, np.argsort(LABEL_DATA))],
)
def test_color_labels_negative_index(negative_indexing, cmap_index):
    labels = pv.ImageData(dimensions=(3, 1, 1))
    labels['data'] = LABEL_DATA
    colored = labels.color_labels('viridis', negative_indexing=negative_indexing)
    color_array = colored.active_scalars

    assert np.array_equal(color_array[0], VIRIDIS_RGB[cmap_index[0]])
    assert np.array_equal(color_array[1], VIRIDIS_RGB[cmap_index[1]])
    assert np.array_equal(color_array[2], VIRIDIS_RGB[cmap_index[2]])


def test_color_labels_scalars(uniform):
    active_before = uniform.active_scalars_name
    for name in uniform.array_names:
        colored = uniform.color_labels(scalars=name)
        assert name in colored.active_scalars_name
    assert uniform.active_scalars_name == active_before

    generic = 'generic'
    for name in uniform.array_names:
        uniform.rename_array(name, generic)
    assert all(name == generic for name in uniform.array_names)

    for name in uniform.array_names:
        colored = uniform.color_labels(scalars=name, preference='point')
        assert generic + '_rgb' in colored.point_data

        colored = uniform.color_labels(scalars=name, preference='cell')
        assert generic + '_rgb' in colored.cell_data

    custom = 'custom'
    colored = uniform.color_labels(output_scalars=custom)
    assert custom in colored.array_names


def test_color_labels_invalid_input(uniform):
    match = 'Coloring mode cannot be set when a color dictionary is specified.'
    with pytest.raises(TypeError, match=match):
        uniform.color_labels({}, coloring_mode='index')

    match = "Colormap 'bwr' must be a ListedColormap, got LinearSegmentedColormap instead."
    with pytest.raises(TypeError, match=match):
        uniform.color_labels('bwr')

    match = 'color must be an instance of'
    with pytest.raises(TypeError, match=match):
        uniform.color_labels([[1]])
    match = (
        'Invalid colors. Colors must be one of:\n'
        '  - sequence of color-like values,\n'
        '  - dict with color-like values,\n'
        '  - named colormap string.'
    )
    with pytest.raises(ValueError, match=match):
        uniform.color_labels('fake')

    match = "Negative indexing is not supported with 'cycle' mode enabled."
    with pytest.raises(ValueError, match=match):
        uniform.color_labels(coloring_mode='cycle', negative_indexing=True)

    match = (
        'Multi-component scalars are not supported for coloring. '
        'Scalar array Normals must be one-dimensional.'
    )
    with pytest.raises(ValueError, match=match):
        pv.Sphere().color_labels(scalars='Normals')


@pytest.mark.parametrize('color_type', ['float_rgb', 'int_rgba'])
def test_color_labels_return_dict(labeled_image, color_type):
    expected_keys = np.unique(labeled_image.active_scalars)

    input_colors = matplotlib_default_colors
    input_keys = list(range(len(input_colors)))
    assert set(expected_keys) < set(input_keys)

    mapping_in = dict(zip(input_keys, input_colors, strict=True))
    colored_mesh, mapping_out = labeled_image.color_labels(
        mapping_in, return_dict=True, color_type=color_type
    )
    assert isinstance(colored_mesh, type(labeled_image))
    assert isinstance(mapping_out, dict)
    assert set(mapping_out.keys()) < set(mapping_in.keys())

    for key in expected_keys:
        expected_color = getattr(pv.Color(mapping_in[key]), color_type)
        actual_color = mapping_out[key]
        assert actual_color == expected_color
