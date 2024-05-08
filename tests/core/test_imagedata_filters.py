import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples

VTK93 = pv.vtk_version_info >= (9, 3)


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    mesh = label_map.contour_labeled()

    assert label_map.point_data.active_scalars.max() == 29
    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_smoothing():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract smooth surface for each label
    mesh = label_map.contour_labeled(smoothing=True)
    # this somehow mutates the object... also the n_labels is likely not correct

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_reduced_labels_count():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    mesh = label_map.contour_labeled(n_labels=2)
    # this somehow mutates the object... also the n_labels is likely not correct

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 2


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_triangle_output_mesh():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    mesh = label_map.contour_labeled(scalars='MetaImage', output_mesh_type='triangles')

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_boundary_output_style():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    mesh = label_map.contour_labeled(output_style='boundary')

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_invalid_output_mesh_type():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    with pytest.raises(ValueError):  # noqa: PT011
        label_map.contour_labeled(output_mesh_type='invalid')


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_invalid_output_style():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    with pytest.raises(NotImplementedError):
        label_map.contour_labeled(output_style='selected')

    with pytest.raises(ValueError):  # noqa: PT011
        label_map.contour_labeled(output_style='invalid')


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_scalars():
    # Load a 3D label map (segmentation of a frog's tissue)
    # and create a new array with reduced number of labels
    label_map = examples.download_frog_tissue()
    label_map['labels'] = label_map['MetaImage'] // 2

    # Extract surface for each label
    mesh = label_map.contour_labeled(scalars='labels')

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 14


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_invalid_scalars():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Nonexistent scalar key
    with pytest.raises(KeyError):
        label_map.contour_labeled(scalars='nonexistent_key')

    # Using cell data
    label_map.cell_data['cell_data'] = np.zeros(label_map.n_cells)
    with pytest.raises(ValueError, match='Can only process point data'):
        label_map.contour_labeled(scalars='cell_data')

    # When no scalas are given and active scalars are not point data
    label_map.set_active_scalars('cell_data', preference='cell')
    with pytest.raises(ValueError, match='active scalars must be point array'):
        label_map.contour_labeled()


@pytest.fixture()
def single_point_image():
    image = pv.ImageData(dimensions=(1, 1, 1))
    image.point_data['image'] = 99
    image.point_data['other'] = 42
    return image


@pytest.mark.parametrize('pad_size', [1, 2])
@pytest.mark.parametrize('value', [-1, 0, 1, 2])
@pytest.mark.parametrize('all_dimensions', [True, False])
def test_pad_image(single_point_image, pad_size, value, all_dimensions):
    image_point_value = single_point_image['image'][0]
    if all_dimensions:
        # Input is expected to be padded
        dim = pad_size * 2 + 1
        expected_dimensions = (dim, dim, dim)
        expected_array = np.ones(np.prod(expected_dimensions)).reshape(*expected_dimensions) * value
        expected_array[pad_size, pad_size, pad_size] = image_point_value
    else:
        # Input is all singletons, expect no padding to be applied
        expected_dimensions = (1, 1, 1)
        expected_array = image_point_value

    padded = single_point_image.pad_image(
        pad_size=pad_size,
        value=value,
        all_dimensions=all_dimensions,
    )
    assert padded.dimensions == expected_dimensions

    # Test correct padding values
    actual_array = padded['image']
    assert actual_array.size == expected_array.size
    assert np.array_equal(actual_array, expected_array.ravel())


@pytest.mark.parametrize(
    ('pad_width', 'expected_dimensions', 'expected_bounds'),
    [
        ((1, 0), (3, 1, 1), (-1, 1, 0, 0, 0, 0)),
        ((0, 1), (1, 3, 1), (0, 0, -1, 1, 0, 0)),
        ((1, 0, 0), (3, 1, 1), (-1, 1, 0, 0, 0, 0)),
        ((0, 1, 0), (1, 3, 1), (0, 0, -1, 1, 0, 0)),
        ((0, 0, 1), (1, 1, 3), (0, 0, 0, 0, -1, 1)),
    ],
)
def test_pad_image_pad_width_axis(
    single_point_image,
    pad_width,
    expected_dimensions,
    expected_bounds,
):
    image_point_value = single_point_image['image'][0]
    pad_value = 7

    padded = single_point_image.pad_image(
        pad_size=pad_width,
        all_dimensions=True,
        value=pad_value,
    )
    assert padded.dimensions == expected_dimensions
    assert padded.bounds == expected_bounds
    assert padded['image'][0] == pad_value
    assert padded['image'][1] == image_point_value
    assert padded['image'][2] == pad_value


@pytest.mark.parametrize(
    ('pad_width', 'expected_dimensions', 'expected_bounds'),
    [
        ((1, 0, 0, 0), (2, 1, 1), (-1, 0, 0, 0, 0, 0)),
        ((0, 1, 0, 0), (2, 1, 1), (0, 1, 0, 0, 0, 0)),
        ((0, 0, 1, 0), (1, 2, 1), (0, 0, -1, 0, 0, 0)),
        ((0, 0, 0, 1), (1, 2, 1), (0, 0, 0, 1, 0, 0)),
        ((1, 0, 0, 0, 0, 0), (2, 1, 1), (-1, 0, 0, 0, 0, 0)),
        ((0, 1, 0, 0, 0, 0), (2, 1, 1), (0, 1, 0, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), (1, 2, 1), (0, 0, -1, 0, 0, 0)),
        ((0, 0, 0, 1, 0, 0), (1, 2, 1), (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 0, 1, 0), (1, 1, 2), (0, 0, 0, 0, -1, 0)),
        ((0, 0, 0, 0, 0, 1), (1, 1, 2), (0, 0, 0, 0, 0, 1)),
    ],
)
def test_pad_image_pad_width_bounds(
    single_point_image,
    pad_width,
    expected_dimensions,
    expected_bounds,
):
    image_point_value = single_point_image['image'][0]
    other_point_value = single_point_image['other'][0]
    pad_value = 7

    padded = single_point_image.pad_image(
        pad_size=pad_width,
        all_dimensions=True,
        value=pad_value,
        all_scalars=True,
    )
    assert single_point_image.active_scalars_name == 'image'
    assert padded.active_scalars_name == 'image'
    assert padded.dimensions == expected_dimensions
    assert padded.bounds == expected_bounds

    if np.any(np.array(expected_bounds) > 0):
        assert padded['image'][0] == image_point_value
        assert padded['image'][1] == pad_value
        assert padded['other'][0] == other_point_value
        assert padded['other'][1] == pad_value
    else:
        assert padded['image'][0] == pad_value
        assert padded['image'][1] == image_point_value
        assert padded['other'][0] == pad_value
        assert padded['other'][1] == other_point_value


@pytest.mark.parametrize('all_scalars', [True, False])
@pytest.mark.parametrize(('scalars', 'expected_scalars'), [(None, 'image'), ('other', 'other')])
def test_pad_image_scalars(single_point_image, all_scalars, scalars, expected_scalars):
    padded = single_point_image.pad_image(0, scalars=scalars, all_scalars=all_scalars)
    assert padded.active_scalars_name == expected_scalars
    actual_array_names = padded.array_names
    if all_scalars:
        assert set(actual_array_names) == {'image', 'other'}
    else:
        assert actual_array_names == [expected_scalars]


@pytest.mark.parametrize('all_point_data', [True, False])
def test_pad_image_does_not_pad_cell_data(uniform, all_point_data):
    assert len(uniform.cell_data.keys()) != 0
    scalars = 'Spatial Point Data'
    padded = uniform.pad_image(all_scalars=all_point_data)
    assert padded.active_scalars_name == scalars
    actual_array_names = padded.array_names
    assert actual_array_names == [scalars]


@pytest.mark.parametrize('value', ['wrap', 'mirror'])
def test_pad_image_wrap_mirror(uniform, value):
    dims = np.array(uniform.dimensions)
    scalars = uniform.active_scalars
    scalars3D = scalars.reshape(dims)
    pad_size = 1

    padded = uniform.pad_image(value, pad_size=pad_size)
    padded_scalars3D = padded.active_scalars.reshape(dims + pad_size * 2)
    if value == 'wrap':
        assert np.array_equal(padded_scalars3D[1:-1, 0, 0], scalars3D[:, -1, -1])
    else:
        assert np.array_equal(padded_scalars3D[1:-1, 0, 0], scalars3D[:, 0, 0])


@pytest.fixture()
def logo():
    return examples.load_logo()


def test_pad_image_multi_component(single_point_image):
    single_point_image.clear_data()
    new_value = np.array([10, 20, 30, 40])
    single_point_image['scalars'] = [new_value]

    dims = np.array(single_point_image.dimensions)
    pad_size = 10

    padded = single_point_image.pad_image(
        new_value,
        pad_size=pad_size,
        all_dimensions=True,
        all_scalars=True,
    )
    assert np.array_equal(len(padded.active_scalars), np.prod(dims + pad_size * 2))
    assert np.all(padded.active_scalars == new_value)

    single_point_image['scalars2'] = [new_value * 2]

    padded = single_point_image.pad_image(
        'wrap',
        pad_size=pad_size,
        all_dimensions=True,
        all_scalars=True,
    )
    assert np.array_equal(len(padded.active_scalars), np.prod(dims + pad_size * 2))
    assert np.all(padded.active_scalars == new_value)
    assert np.all(padded['scalars2'] == new_value * 2)


def test_pad_image_raises(single_point_image, uniform, logo):
    match = 'Pad size cannot be negative.'
    with pytest.raises(ValueError, match=match):
        single_point_image.pad_image(pad_size=-1)

    match = "Pad size must have 1, 2, 3, 4, or 6 values, got 5 instead."
    with pytest.raises(ValueError, match=match):
        single_point_image.pad_image(pad_size=(1, 2, 3, 4, 5))

    match = 'Pad size must be one dimensional. Got 2 dimensions.'
    with pytest.raises(ValueError, match=match):
        single_point_image.pad_image(pad_size=[[1]])

    match = "Pad size must be integers. Got dtype float64."
    with pytest.raises(TypeError, match=match):
        single_point_image.pad_image(pad_size=1.0)

    match = "Scalars 'Spatial Cell Data' must be associated with point data. Got cell data instead."
    with pytest.raises(ValueError, match=match):
        uniform.pad_image(scalars='Spatial Cell Data')

    match = "Input value 0.1 with dtype 'float64' is not compatible with dtype 'uint8' of array PNGImage."
    with pytest.raises(TypeError, match=re.escape(match)):
        logo.pad_image(0.1)

    match = "Invalid padding value foo. Must be 'mirror' or 'wrap', or a number/component vector for constant padding."
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image('foo')

    match = "Invalid padding value [[2]]. Must be 'mirror' or 'wrap', or a number/component vector for constant padding."
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image([[2]])

    match = "Number of components (2) in padding value (0, 0) must match the number components (4) in array 'PNGImage'."
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image((0, 0))

    logo['single'] = range(logo.n_points)  # Create data with varying num array components
    match = (
        "Cannot pad array 'single' with value (0, 0, 0, 0). Number of components (1) in 'single' must match the number of components (4) in value."
        "\nTry setting `all_scalars=False` or update the array."
    )
    logo.pad_image(value=(0, 0, 0, 0), all_scalars=False)
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image(value=(0, 0, 0, 0), all_scalars=True)
