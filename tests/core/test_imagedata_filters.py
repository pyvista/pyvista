from __future__ import annotations

import operator
import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import PyVistaDeprecationWarning

VTK93 = pv.vtk_version_info >= (9, 3)


@pytest.fixture
def logo():
    return examples.load_logo()


def variable_dimensionality_image(dimensions):
    image = pv.ImageData(dimensions=dimensions)
    image.point_data['image'] = 99
    image.point_data['other'] = 42
    image.cell_data['data'] = 142
    return image


@pytest.fixture
def zero_dimensionality_image():
    return variable_dimensionality_image(dimensions=(1, 1, 1))


@pytest.fixture
def one_dimensionality_image():
    return variable_dimensionality_image(dimensions=(1, 2, 1))


@pytest.fixture
def two_dimensionality_image():
    return variable_dimensionality_image(dimensions=(2, 1, 2))


@pytest.fixture
def three_dimensionality_image():
    return variable_dimensionality_image(dimensions=(2, 2, 2))


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.load_frog_tissues()

    # Extract surface for each label
    mesh = label_map.contour_labeled()

    assert label_map.point_data.active_scalars.max() == 29
    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_smoothing():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.load_frog_tissues()

    # Extract smooth surface for each label
    mesh = label_map.contour_labeled(smoothing=True)
    # this somehow mutates the object... also the n_labels is likely not correct

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_reduced_labels_count():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.load_frog_tissues()

    # Extract surface for each label
    mesh = label_map.contour_labeled(n_labels=2)
    # this somehow mutates the object... also the n_labels is likely not correct

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 2


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_triangle_output_mesh():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.load_frog_tissues()

    # Extract surface for each label
    mesh = label_map.contour_labeled(scalars='MetaImage', output_mesh_type='triangles')

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_boundary_output_style():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.load_frog_tissues()

    # Extract surface for each label
    mesh = label_map.contour_labeled(output_style='boundary')

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_invalid_output_mesh_type():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.load_frog_tissues()

    # Extract surface for each label
    with pytest.raises(ValueError):  # noqa: PT011
        label_map.contour_labeled(output_mesh_type='invalid')


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_invalid_output_style():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.load_frog_tissues()

    # Extract surface for each label
    with pytest.raises(NotImplementedError):
        label_map.contour_labeled(output_style='selected')

    with pytest.raises(ValueError):  # noqa: PT011
        label_map.contour_labeled(output_style='invalid')


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_scalars():
    # Load a 3D label map (segmentation of a frog's tissue)
    # and create a new array with reduced number of labels
    label_map = examples.load_frog_tissues()
    label_map['labels'] = label_map['MetaImage'] // 2

    # Extract surface for each label
    mesh = label_map.contour_labeled(scalars='labels')

    assert 'BoundaryLabels' in mesh.cell_data
    assert np.max(mesh['BoundaryLabels'][:, 0]) == 14


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_invalid_scalars():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.load_frog_tissues()

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


@pytest.fixture
def uniform_many_scalars(uniform):
    uniform['Spatial Point Data2'] = uniform['Spatial Point Data'] * 2
    uniform['Spatial Cell Data2'] = uniform['Spatial Cell Data'] * 2
    return uniform


@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize(
    'active_scalars',
    [None, 'Spatial Point Data2', 'Spatial Point Data'],
)
def test_points_to_cells(uniform_many_scalars, active_scalars, copy):
    uniform_many_scalars.set_active_scalars(active_scalars)

    point_voxel_image = uniform_many_scalars
    point_voxel_points = point_voxel_image.points

    cell_voxel_image = point_voxel_image.points_to_cells(copy=copy)
    cell_voxel_center_points = cell_voxel_image.cell_centers().points

    assert point_voxel_image.n_points == cell_voxel_image.n_cells
    assert cell_voxel_image.active_scalars_name == active_scalars
    assert set(cell_voxel_image.array_names) == {'Spatial Point Data', 'Spatial Point Data2'}
    assert np.array_equal(point_voxel_points, cell_voxel_center_points)
    assert np.array_equal(point_voxel_image.active_scalars, cell_voxel_image.active_scalars)
    assert cell_voxel_image.point_data.keys() == []

    for array_in, array_out in zip(
        point_voxel_image.point_data.keys(),
        cell_voxel_image.cell_data.keys(),
    ):
        shares_memory = np.shares_memory(point_voxel_image[array_in], cell_voxel_image[array_out])
        assert not shares_memory if copy else shares_memory


@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize(
    'active_scalars',
    [None, 'Spatial Cell Data2', 'Spatial Cell Data'],
)
def test_cells_to_points(uniform_many_scalars, active_scalars, copy):
    uniform_many_scalars.set_active_scalars(active_scalars)

    cell_voxel_image = uniform_many_scalars
    cell_voxel_center_points = cell_voxel_image.cell_centers().points

    point_voxel_image = cell_voxel_image.cells_to_points(copy=copy)
    point_voxel_points = point_voxel_image.points

    assert cell_voxel_image.n_cells == point_voxel_image.n_points
    assert cell_voxel_image.active_scalars_name == active_scalars
    assert set(point_voxel_image.array_names) == {'Spatial Cell Data', 'Spatial Cell Data2'}
    assert np.array_equal(cell_voxel_center_points, point_voxel_points)
    assert np.array_equal(cell_voxel_image.active_scalars, point_voxel_image.active_scalars)
    assert point_voxel_image.cell_data.keys() == []

    for array_in, array_out in zip(
        cell_voxel_image.cell_data.keys(),
        point_voxel_image.point_data.keys(),
    ):
        shares_memory = np.shares_memory(cell_voxel_image[array_in], point_voxel_image[array_out])
        assert not shares_memory if copy else shares_memory


def test_points_to_cells_scalars(uniform):
    scalars = 'Spatial Point Data'
    converted = uniform.points_to_cells(scalars)
    assert converted.active_scalars_name == scalars
    assert converted.cell_data.keys() == [scalars]

    match = "Scalars 'Spatial Cell Data' must be associated with point data. Got cell data instead."
    with pytest.raises(ValueError, match=match):
        uniform.points_to_cells('Spatial Cell Data')


def test_cells_to_points_scalars(uniform):
    scalars = 'Spatial Cell Data'
    converted = uniform.cells_to_points(scalars)
    assert converted.active_scalars_name == scalars
    assert converted.point_data.keys() == [scalars]

    match = (
        "Scalars 'Spatial Point Data' must be associated with cell data. Got point data instead."
    )
    with pytest.raises(ValueError, match=match):
        uniform.cells_to_points('Spatial Point Data')


def test_points_to_cells_and_cells_to_points_dimensions(
    uniform,
    logo,
    zero_dimensionality_image,
    one_dimensionality_image,
    two_dimensionality_image,
    three_dimensionality_image,
):
    assert uniform.dimensions == (10, 10, 10)
    assert uniform.points_to_cells().dimensions == (11, 11, 11)
    assert uniform.cells_to_points().dimensions == (9, 9, 9)
    assert uniform.points_to_cells(dimensionality='preserve').dimensions == (11, 11, 11)
    assert uniform.cells_to_points(dimensionality='preserve').dimensions == (9, 9, 9)

    assert logo.dimensions == (1920, 718, 1)
    assert logo.points_to_cells().dimensions == (1921, 719, 1)
    assert logo.cells_to_points().dimensions == (1919, 717, 1)
    assert logo.points_to_cells(dimensionality='preserve').dimensions == (1921, 719, 1)
    assert logo.cells_to_points(dimensionality='preserve').dimensions == (1919, 717, 1)

    assert zero_dimensionality_image.dimensions == (1, 1, 1)
    assert zero_dimensionality_image.points_to_cells(
        dimensionality=(True, True, True)
    ).dimensions == (2, 2, 2)
    assert zero_dimensionality_image.points_to_cells(
        dimensionality=(True, True, False)
    ).dimensions == (2, 2, 1)
    assert zero_dimensionality_image.points_to_cells(
        dimensionality=(False, False, False)
    ).dimensions == (1, 1, 1)
    assert zero_dimensionality_image.points_to_cells(dimensionality='0D').dimensions == (1, 1, 1)
    assert zero_dimensionality_image.points_to_cells(dimensionality='1D').dimensions == (2, 1, 1)
    assert zero_dimensionality_image.points_to_cells(dimensionality='2D').dimensions == (2, 2, 1)
    assert zero_dimensionality_image.points_to_cells(dimensionality='3D').dimensions == (2, 2, 2)
    assert zero_dimensionality_image.cells_to_points(dimensionality='0D').dimensions == (1, 1, 1)

    assert one_dimensionality_image.dimensions == (1, 2, 1)
    assert one_dimensionality_image.points_to_cells(dimensionality='1D').dimensions == (1, 3, 1)
    assert one_dimensionality_image.points_to_cells(dimensionality='2D').dimensions == (2, 3, 1)
    assert one_dimensionality_image.points_to_cells(dimensionality='3D').dimensions == (2, 3, 2)
    assert one_dimensionality_image.cells_to_points(dimensionality='0D').dimensions == (1, 1, 1)
    assert one_dimensionality_image.points_to_cells(dimensionality='1D').cells_to_points(
        dimensionality='1D'
    ).dimensions == (1, 2, 1)

    assert two_dimensionality_image.dimensions == (2, 1, 2)
    assert two_dimensionality_image.points_to_cells(dimensionality='2D').dimensions == (3, 1, 3)
    assert two_dimensionality_image.points_to_cells(dimensionality='3D').dimensions == (3, 2, 3)
    assert two_dimensionality_image.cells_to_points(dimensionality='0D').dimensions == (1, 1, 1)
    assert two_dimensionality_image.points_to_cells(dimensionality='2D').cells_to_points(
        dimensionality='2D'
    ).dimensions == (2, 1, 2)

    assert three_dimensionality_image.dimensions == (2, 2, 2)
    assert three_dimensionality_image.points_to_cells(dimensionality='3D').dimensions == (3, 3, 3)
    assert three_dimensionality_image.cells_to_points(dimensionality='0D').dimensions == (1, 1, 1)
    assert three_dimensionality_image.points_to_cells(dimensionality='3D').cells_to_points(
        dimensionality='3D'
    ).dimensions == (2, 2, 2)


def test_points_to_cells_and_cells_to_points_dimensions_incorrect_number_data():
    image = pv.ImageData(dimensions=(1, 2, 2))
    with pytest.raises(
        ValueError,
        match=(
            r'Cannot re-mesh points to cells. The dimensions of the input \(1, 2, 2\) is not compatible'
            r' with the dimensions of the output \(2, 2, 2\) and would require to map 4 points on 1 cells.'
        ),
    ):
        image.points_to_cells(dimensionality=[True, False, False])
    with pytest.raises(
        ValueError,
        match=(
            r'Cannot re-mesh cells to points. The dimensions of the input \(1, 2, 2\) is not compatible'
            r' with the dimensions of the output \(1, 2, 2\) and would require to map 1 cells on 4 points.'
        ),
    ):
        image.cells_to_points(dimensionality='2D')


@pytest.mark.parametrize('pad_size', [1, 2])
@pytest.mark.parametrize('pad_value', [-1, 0, 1, 2])
@pytest.mark.parametrize('dimensionality', [(True, True, True), (False, False, False)])
def test_pad_image(zero_dimensionality_image, pad_size, pad_value, dimensionality):
    image_point_value = zero_dimensionality_image['image'][0]
    if all(dimensionality):
        # Input is expected to be padded
        dim = pad_size * 2 + 1
        expected_dimensions = (dim, dim, dim)
        expected_array = (
            np.ones(np.prod(expected_dimensions)).reshape(*expected_dimensions) * pad_value
        )
        expected_array[pad_size, pad_size, pad_size] = image_point_value
    else:
        # Input is all singletons, expect no padding to be applied
        expected_dimensions = (1, 1, 1)
        expected_array = image_point_value

    padded = zero_dimensionality_image.pad_image(
        pad_size=pad_size,
        pad_value=pad_value,
        dimensionality=dimensionality,
    )
    assert padded.dimensions == expected_dimensions

    # Test correct padding values
    actual_array = padded['image']
    assert actual_array.size == expected_array.size
    assert np.array_equal(actual_array, expected_array.ravel())


@pytest.mark.parametrize(
    ('pad_size', 'expected_dimensions', 'expected_bounds'),
    [
        ((1, 0), (3, 1, 1), (-1, 1, 0, 0, 0, 0)),
        ((0, 1), (1, 3, 1), (0, 0, -1, 1, 0, 0)),
        ((1, 0, 0), (3, 1, 1), (-1, 1, 0, 0, 0, 0)),
        ((0, 1, 0), (1, 3, 1), (0, 0, -1, 1, 0, 0)),
        ((0, 0, 1), (1, 1, 3), (0, 0, 0, 0, -1, 1)),
    ],
)
def test_pad_image_pad_size_axis(
    zero_dimensionality_image,
    pad_size,
    expected_dimensions,
    expected_bounds,
):
    image_point_value = zero_dimensionality_image['image'][0]
    pad_value = 7

    padded = zero_dimensionality_image.pad_image(
        pad_size=pad_size,
        dimensionality=(True, True, True),
        pad_value=pad_value,
    )
    assert padded.dimensions == expected_dimensions
    assert padded.bounds == expected_bounds
    assert padded['image'][0] == pad_value
    assert padded['image'][1] == image_point_value
    assert padded['image'][2] == pad_value


@pytest.mark.parametrize(
    ('pad_size', 'expected_dimensions', 'expected_bounds'),
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
def test_pad_image_pad_size_bounds(
    zero_dimensionality_image,
    pad_size,
    expected_dimensions,
    expected_bounds,
):
    image_point_value = zero_dimensionality_image['image'][0]
    other_point_value = zero_dimensionality_image['other'][0]
    pad_value = 7

    padded = zero_dimensionality_image.pad_image(
        pad_size=pad_size,
        dimensionality=(True, True, True),
        pad_value=pad_value,
        pad_all_scalars=True,
    )
    assert zero_dimensionality_image.active_scalars_name == 'image'
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
def test_pad_image_scalars(zero_dimensionality_image, all_scalars, scalars, expected_scalars):
    padded = zero_dimensionality_image.pad_image(0, scalars=scalars, pad_all_scalars=all_scalars)
    assert padded.active_scalars_name == expected_scalars
    actual_array_names = padded.array_names
    if all_scalars:
        assert set(actual_array_names) == {'image', 'other'}
    else:
        assert actual_array_names == [expected_scalars]


@pytest.mark.parametrize('all_scalars', [True, False])
def test_pad_image_does_not_pad_cell_data(uniform, all_scalars):
    assert len(uniform.cell_data.keys()) != 0
    scalars = 'Spatial Point Data'
    padded = uniform.pad_image(pad_all_scalars=all_scalars)
    assert padded.active_scalars_name == scalars
    actual_array_names = padded.array_names
    assert actual_array_names == [scalars]


@pytest.mark.parametrize('pad_value', ['wrap', 'mirror'])
def test_pad_image_wrap_mirror(uniform, pad_value):
    dims = np.array(uniform.dimensions)
    scalars = uniform.active_scalars
    scalars3D = scalars.reshape(dims)
    pad_size = 1

    padded = uniform.pad_image(pad_value, pad_size=pad_size)
    padded_scalars3D = padded.active_scalars.reshape(dims + pad_size * 2)
    if pad_value == 'wrap':
        assert np.array_equal(padded_scalars3D[1:-1, 0, 0], scalars3D[:, -1, -1])
    else:
        assert np.array_equal(padded_scalars3D[1:-1, 0, 0], scalars3D[:, 0, 0])


def test_pad_image_multi_component(zero_dimensionality_image):
    zero_dimensionality_image.clear_data()
    new_value = np.array([10, 20, 30, 40])
    zero_dimensionality_image['scalars'] = [new_value]

    dims = np.array(zero_dimensionality_image.dimensions)
    pad_size = 10

    padded = zero_dimensionality_image.pad_image(
        new_value,
        pad_size=pad_size,
        dimensionality=(True, True, True),
        pad_all_scalars=True,
        progress_bar=True,
    )
    assert np.array_equal(len(padded.active_scalars), np.prod(dims + pad_size * 2))
    assert np.all(padded.active_scalars == new_value)

    zero_dimensionality_image['scalars2'] = [new_value * 2]

    padded = zero_dimensionality_image.pad_image(
        'wrap',
        pad_size=pad_size,
        dimensionality=(True, True, True),
        pad_all_scalars=True,
    )
    assert np.array_equal(len(padded.active_scalars), np.prod(dims + pad_size * 2))
    assert np.all(padded.active_scalars == new_value)
    assert np.all(padded['scalars2'] == new_value * 2)


def test_pad_image_raises(zero_dimensionality_image, uniform, logo):
    match = 'Pad size cannot be negative. Got -1.'
    with pytest.raises(ValueError, match=match):
        zero_dimensionality_image.pad_image(pad_size=-1)

    match = 'Pad size must have 1, 2, 3, 4, or 6 values, got 5 instead.'
    with pytest.raises(ValueError, match=match):
        zero_dimensionality_image.pad_image(pad_size=(1, 2, 3, 4, 5))

    match = 'Pad size must be one dimensional. Got 2 dimensions.'
    with pytest.raises(ValueError, match=match):
        zero_dimensionality_image.pad_image(pad_size=[[1]])

    match = 'Pad size must be integers. Got dtype float64.'
    with pytest.raises(TypeError, match=match):
        zero_dimensionality_image.pad_image(pad_size=1.0)

    match = "Scalars 'Spatial Cell Data' must be associated with point data. Got cell data instead."
    with pytest.raises(ValueError, match=match):
        uniform.pad_image(scalars='Spatial Cell Data')

    match = (
        "Pad value 0.1 with dtype 'float64' is not compatible with dtype 'uint8' of array PNGImage."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        logo.pad_image(0.1)

    match = "Invalid pad value foo. Must be 'mirror' or 'wrap', or a number/component vector for constant padding."
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image('foo')

    match = "Invalid pad value [[2]]. Must be 'mirror' or 'wrap', or a number/component vector for constant padding."
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image([[2]])

    match = "Number of components (2) in pad value (0, 0) must match the number components (4) in array 'PNGImage'."
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image((0, 0))

    logo['single'] = range(logo.n_points)  # Create data with varying num array components
    match = (
        "Cannot pad array 'single' with value (0, 0, 0, 0). Number of components (1) in 'single' must match the number of components (4) in value."
        '\nTry setting `pad_all_scalars=False` or update the array.'
    )
    logo.pad_image(pad_value=(0, 0, 0, 0), pad_all_scalars=False)
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image(pad_value=(0, 0, 0, 0), pad_all_scalars=True)


def test_pad_image_deprecation(zero_dimensionality_image):
    match = 'Use of `pad_singleton_dims=True` is deprecated. Use `dimensionality="3D"` instead'
    with pytest.warns(PyVistaDeprecationWarning, match=match):
        zero_dimensionality_image.pad_image(pad_value=1, pad_singleton_dims=True)
        if pv._version.version_info >= (0, 47):
            raise RuntimeError('Passing `pad_singleton_dims` should raise an error.')
        if pv._version.version_info >= (0, 48):
            raise RuntimeError('Remove `pad_singleton_dims`.')

    match = (
        'Use of `pad_singleton_dims=False` is deprecated. Use `dimensionality="preserve"` instead'
    )
    with pytest.warns(PyVistaDeprecationWarning, match=match):
        zero_dimensionality_image.pad_image(pad_value=1, pad_singleton_dims=False)
        if pv._version.version_info >= (0, 47):
            raise RuntimeError('Passing `pad_singleton_dims` should raise an error.')
        if pv._version.version_info >= (0, 48):
            raise RuntimeError('Remove `pad_singleton_dims`.')


@pytest.fixture
def segmented_grid():
    segmented_grid = pv.ImageData(dimensions=(4, 3, 3))
    segmented_grid.cell_data['Data'] = [0, 0, 0, 1, 0, 1, 1, 2, 0, 0, 0, 0]
    return segmented_grid


def test_label_connectivity(segmented_grid):
    # Test default parameters
    connected, labels, sizes = segmented_grid.label_connectivity(scalar_range='foreground')
    assert isinstance(connected, pv.ImageData)
    assert connected.bounds == segmented_grid.bounds
    assert 'RegionId' in connected.cell_data
    assert connected.cell_data['RegionId'].dtype == 'uint8'
    # Test that three distinct connected regions were labelled
    assert all(labels == [1, 2, 3])
    # Test that the first region id corresponds to the largest region (2 cells)
    assert sizes[0] == 2


def test_label_connectivity_point_data(segmented_grid):
    # Test default parameters
    segmented_points = segmented_grid.cells_to_points()
    connected, labels, sizes = segmented_points.label_connectivity(scalar_range='foreground')
    assert isinstance(connected, pv.ImageData)
    assert connected.bounds == segmented_points.bounds
    assert 'RegionId' in connected.point_data
    assert connected.point_data['RegionId'].dtype == 'uint8'
    # Test that three distinct connected regions were labelled
    assert all(labels == [1, 2, 3])


def test_label_connectivity_scalar(segmented_grid):
    segmented_grid.cell_data['AdditionalData'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    segmented_grid.set_active_scalars(name='AdditionalData')
    connected, labels, sizes = segmented_grid.label_connectivity(
        scalars='Data', scalar_range='foreground'
    )
    assert all(labels == [1, 2, 3])


def test_label_connectivity_largest_region(segmented_grid):
    connected, labels, sizes = segmented_grid.label_connectivity(
        scalar_range='foreground', extraction_mode='largest'
    )
    # Test that only one region was labelled
    assert all(labels == [1])
    # Test that the first region id corresponds to the largest region (2 cells)
    assert sizes[0] == 2


def test_label_connectivity_seed_points(segmented_grid):
    points = [(2, 1, 0), (0, 0, 1)]
    with pytest.warns(
        UserWarning,
        match='Points is not a float type. This can cause'
        ' issues when transforming or applying filters. Casting to ``np.float32``.'
        ' Disable this by passing ``force_float=False``.',
    ):
        connected, labels, sizes = segmented_grid.label_connectivity(
            scalar_range='foreground',
            extraction_mode='seeded',
            point_seeds=points,
            label_mode='seeds',
        )
    # Test that two regions were labelled
    assert all(labels == [1, 2])
    # Test that the second region id corresponds to the largest region (2 cells) which
    # corresponds to the second seed coordinates
    assert sizes[1] == 2


def test_label_connectivity_seed_points_vtkDataSet(segmented_grid):
    points = pv.PolyData()
    points.points = [(2, 1, 0), (0, 0, 1)]
    connected, labels, sizes = segmented_grid.label_connectivity(
        scalar_range='foreground',
        extraction_mode='seeded',
        point_seeds=points,
        label_mode='seeds',
    )
    # Test that two regions were labelled
    assert all(labels == [1, 2])
    # Test that the second region id corresponds to the largest region (2 cells) which
    # corresponds to the second seed coordinates
    assert sizes[1] == 2


def test_label_connectivity_scalar_range_whole_number(segmented_grid):
    # Exclude the cell with a 2 value
    connected, labels, sizes = segmented_grid.label_connectivity(scalar_range=[1, 1])
    # Test that three distinct connected regions were labelled
    assert all(labels == [1, 2, 3])
    # Test that the first region id has 1 cell
    assert sizes[0] == 1


def test_label_connectivity_scalar_range_fractional_number(segmented_grid):
    # Exclude the cell with a 2 value
    connected, labels, sizes = segmented_grid.label_connectivity(scalar_range=[0.5, 1.5])
    # Test that three distinct connected regions were labelled
    assert all(labels == [1, 2, 3])
    # Test that the first region id has 1 cell
    assert sizes[0] == 1


def test_label_connectivity_auto_scalar_range(segmented_grid):
    # Exclude the cell with a 2 value
    connected, labels, sizes = segmented_grid.label_connectivity(scalar_range='auto')
    # Test that only one connected regions was labelled
    assert all(labels == 1)
    # Test that the region has 12 cell
    assert sizes[0] == 12


def test_label_connectivity_scalar_range_default_vtk(segmented_grid):
    connected, labels, sizes = segmented_grid.label_connectivity(
        scalar_range='vtk_default', inplace=True
    )
    # Test that three distinct connected regions were labelled
    assert all(labels == [1, 2, 3])
    # Test that float are re-casted to int for inplace operation
    assert np.issubdtype(connected.cell_data['Data'].dtype, np.integer)


def test_label_connectivity_constant_label(segmented_grid):
    connected, labels, sizes = segmented_grid.label_connectivity(
        label_mode='constant', constant_value=10
    )
    assert all(l in (0, 10) for l in labels)


def test_label_connectivity_inplace_with_float_casting(segmented_grid):
    segmented_points = segmented_grid.cells_to_points()
    connected, labels, sizes = segmented_grid.label_connectivity(
        inplace=True, scalar_range=[0.5, 2.5]
    )
    assert connected == segmented_grid
    assert 'RegionId' in connected.cell_data
    assert np.issubdtype(connected.cell_data['Data'].dtype, np.integer)

    connected, labels, sizes = segmented_points.label_connectivity(
        inplace=True, scalar_range=[0.5, 2.5]
    )
    assert connected == segmented_points
    assert 'RegionId' in connected.point_data
    assert np.issubdtype(connected.point_data['Data'].dtype, np.integer)


def test_label_connectivity_invalid_parameters(segmented_grid):
    with pytest.raises(
        ValueError,
        match='Invalid `extraction_mode` "invalid", use "all", "largest", or "seeded".',
    ):
        _ = segmented_grid.label_connectivity(extraction_mode='invalid')
    with pytest.raises(
        ValueError, match='`point_seeds` must be specified when `extraction_mode="seeded"`.'
    ):
        _ = segmented_grid.label_connectivity(extraction_mode='seeded')
    with pytest.raises(
        IndexError,
        match=r'tuple index out of range|Given points must be convertible to a numerical array',
    ):
        _ = segmented_grid.label_connectivity(extraction_mode='seeded', point_seeds=2.0)
    with pytest.raises(
        ValueError, match='Invalid `label_mode` "invalid", use "size", "constant", or "seeds".'
    ):
        _ = segmented_grid.label_connectivity(label_mode='invalid')
    with pytest.raises(
        ValueError, match='`point_seeds` must be specified when `label_mode="seeds"`.'
    ):
        _ = segmented_grid.label_connectivity(label_mode='seeds')
    with pytest.raises(
        ValueError, match='Data Range with 2 elements must be sorted in ascending order'
    ):
        _ = segmented_grid.label_connectivity(scalar_range=[2.0, 1.0])
    with pytest.raises(ValueError, match='Shape must be 2'):
        _ = segmented_grid.label_connectivity(scalar_range=[1.0, 2.0, 3.0])
    with pytest.raises(
        ValueError,
        match='`constant_value` must be provided when `extraction_mode`is "constant".',
    ):
        _ = segmented_grid.label_connectivity(label_mode='constant')


@pytest.mark.parametrize(
    ('image_dims', 'operation_mask', 'operator', 'expected_dims_mask', 'expected_dims_result'),
    [
        ((1, 1, 1), (True, True, True), operator.add, True, (2, 4, 6)),
        ((1, 1, 1), (False, False, False), operator.add, False, (1, 1, 1)),
        ((1, 1, 1), (True, False, True), operator.add, (True, False, True), (2, 1, 6)),
        ((1, 1, 1), 'preserve', operator.add, False, (1, 1, 1)),
        ((1, 1, 1), '0D', operator.add, False, (1, 1, 1)),
        ((1, 1, 1), '1D', operator.add, (True, False, False), (2, 1, 1)),
        ((1, 1, 2), '1D', operator.add, (False, False, True), (1, 1, 7)),
        ((1, 1, 1), '2D', operator.add, (True, True, False), (2, 4, 1)),
        ((1, 1, 2), '2D', operator.add, (True, False, True), (2, 1, 7)),
        ((1, 2, 2), '2D', operator.add, (False, True, True), (1, 5, 7)),
        ((1, 1, 1), '3D', operator.add, (True, True, True), (2, 4, 6)),
        ((10, 10, 10), '3D', operator.sub, (True, True, True), (9, 7, 5)),
    ],
)
def test_validate_dim_operation(
    image_dims, operation_mask, operator, expected_dims_mask, expected_dims_result
):
    image = pv.ImageData(dimensions=image_dims)
    dims_mask, dims_result = image._validate_dimensional_operation(
        operation_mask=operation_mask, operator=operator, operation_size=(1, 3, 5)
    )
    assert (dims_mask == expected_dims_mask).all()
    assert (dims_result == expected_dims_result).all()


@pytest.mark.parametrize(
    ('image_dims', 'operation_mask', 'operator', 'error', 'error_message'),
    [
        (
            (1, 1, 1),
            'invalid',
            operator.add,
            ValueError,
            '`invalid` is not a valid `operation_mask`. Use one of [0, 1, 2, 3, "0D", "1D", "2D", "3D", "preserve"].',
        ),
        (
            (1, 1, 1),
            (True, True, True, True),
            operator.add,
            ValueError,
            'Array has shape (4,) which is not allowed.',
        ),
        (
            (1, 1, 1),
            True,
            operator.add,
            ValueError,
            'Array has shape () which is not allowed. Shape must be one of [(3,), (1, 3), (3, 1)]',
        ),
        (
            (2, 2, 2),
            '1D',
            operator.add,
            ValueError,
            'The operation requires to add at least [1 3 5] dimension(s) to (2, 2, 2). A 1D ImageData with dims (>1, 1, 1) cannot be obtained.',
        ),
        (
            (2, 1, 2),
            '3D',
            operator.sub,
            ValueError,
            'The operation requires to sub at least [1 3 5] dimension(s) to (2, 1, 2). A 3D ImageData with dims (>1, >1, >1) cannot be obtained.',
        ),
        (
            (1, 2, 5),
            (True, False, True),
            operator.sub,
            ValueError,
            'The mask (True, False, True), size [1 3 5], and operation sub would result in [0 2 0] which contains <= 0 dimensions.',
        ),
    ],
)
def test_validate_dim_operation_invalid_parameters(
    image_dims, operation_mask, operator, error, error_message
):
    image = pv.ImageData(dimensions=image_dims)
    with pytest.raises(error, match=re.escape(error_message)):
        image._validate_dimensional_operation(
            operation_mask=operation_mask, operator=operator, operation_size=(1, 3, 5)
        )
