import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import PyVistaDeprecationWarning, examples

SURFACE_LABELS = 'surface_labels'
BOUNDARY_LABELS = 'boundary_labels'


def test_contour_labeled_deprecated():
    match = 'This filter produces unexpected results and is deprecated.'
    with pytest.raises(PyVistaDeprecationWarning, match=match):
        pv.ImageData().contour_labeled()


@pytest.fixture()
def labeled_image():
    # Create 4x3x3 image with two adjacent labels

    # First label:
    #   has a single point near center of image,
    #   is adjacent to second label,
    #   is otherwise surrounded by background,

    # Second label:
    #   has two points near center of image,
    #   is adjacent to first label,
    #   has one side touching image boundary,
    #   is otherwise surrounded by background

    dim = (4, 3, 3)
    labels = np.zeros(np.prod(dim))
    labels[17] = 2  # First label
    labels[[18, 19]] = 5  # Second label
    image = pv.ImageData(dimensions=dim)
    image.point_data['labels'] = labels

    label_ids = np.unique(image.point_data.active_scalars).tolist()
    assert label_ids == [0, 2, 5]
    return image


def _get_label_ids(array):
    """Get unique foreground label ids."""
    ids = np.unique(array).tolist()
    ids.remove(0) if 0 in ids else None
    return ids


def _get_ids_with_background_boundary(mesh):
    """Return region ids from the boundary_labels array which share a boundary with the background."""
    extracted = mesh.extract_values(0, component_mode=1)
    external_polygons = extracted[BOUNDARY_LABELS] if BOUNDARY_LABELS in extracted.cell_data else []
    return _get_label_ids(external_polygons)


def _get_ids_with_internal_boundary(mesh):
    """Return region ids from the boundary_labels array which share internal boundaries."""
    extracted = mesh.extract_values(0, invert=True, component_mode=1)
    internal_polygons = extracted[BOUNDARY_LABELS] if BOUNDARY_LABELS in extracted.cell_data else []
    return _get_label_ids(internal_polygons)


@pytest.mark.parametrize('smoothing', [True, False, None])
@pytest.mark.parametrize('output_mesh_type', ['triangles', 'quads'])
@pytest.mark.parametrize('scalars', ['labels', None])
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_scalars_smoothing_output_mesh_type(
    labeled_image,
    smoothing,
    output_mesh_type,
    scalars,
):
    # Determine expected output
    if output_mesh_type == 'triangles' or output_mesh_type is None and smoothing:
        expected_celltype = pv.CellType.TRIANGLE
        multiplier = 2  # quads are subdivided into 2 triangles
    else:
        assert output_mesh_type == 'quads' or not smoothing
        expected_celltype = pv.CellType.QUAD
        multiplier = 1

    # Do test
    mesh = labeled_image.contour_labels(
        scalars=scalars,
        smoothing=smoothing,
        output_mesh_type=output_mesh_type,
        output_labels='boundary',
        duplicate_polygons=False,
    )
    assert BOUNDARY_LABELS in mesh.cell_data
    assert all(cell.type == expected_celltype for cell in mesh.cell)
    label_ids = _get_ids_with_background_boundary(mesh)
    assert label_ids == [2, 5]

    if smoothing:
        assert mesh.area < 0.01
    else:
        assert mesh.area == mesh.n_cells / multiplier


@pytest.mark.parametrize('select_inputs', [None, 2, 5, [2, 5]])
@pytest.mark.parametrize(
    'select_outputs',
    [None, 2, 5, [2, 5]],
)
@pytest.mark.parametrize('internal_polygons', [True, False])
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_boundaries(
    labeled_image,
    select_inputs,
    select_outputs,
    internal_polygons,
):
    # Test correct boundary_labels output
    ALL_LABEL_IDS = np.array([2, 5])

    mesh = labeled_image.contour_labels(
        select_inputs=select_inputs,
        select_outputs=select_outputs,
        internal_polygons=internal_polygons,
        output_labels='boundary',
        duplicate_polygons=False,
    )  # , image_boundary_faces=image_boundary_faces)
    assert BOUNDARY_LABELS in mesh.cell_data
    actual_output_ids = _get_ids_with_background_boundary(mesh)

    # Make sure param values are iterable
    select_inputs_iter = np.atleast_1d(select_inputs) if select_inputs else ALL_LABEL_IDS
    select_outputs_iter = np.atleast_1d(select_outputs) if select_outputs else ALL_LABEL_IDS

    # All selected outputs are expected if it's also selected at the input
    expected_output_ids = [id_ for id_ in select_outputs_iter if id_ in select_inputs_iter]

    assert actual_output_ids == expected_output_ids

    if internal_polygons and len(select_inputs_iter) == 2:
        # The two labels share a boundary by default
        # Boundary exists if no labels removed from input
        expected_shared_region_ids = ALL_LABEL_IDS
        shared_cells = mesh.extract_values([[2, 5], [5, 2]], component_mode='multi')
        assert shared_cells.n_cells == 2
    else:
        expected_shared_region_ids = []

    actual_shared_region_ids = _get_ids_with_internal_boundary(mesh)
    assert np.array_equal(actual_shared_region_ids, expected_shared_region_ids)

    # Make sure temp array created for select_inputs is removed
    assert labeled_image.array_names == ['labels']


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_image_boundaries(labeled_image):
    mesh_true = labeled_image.contour_labels(closed_boundary=True, output_mesh_type='quads')
    mesh_false = labeled_image.contour_labels(closed_boundary=False, output_mesh_type='quads')
    assert mesh_true.n_cells - mesh_false.n_cells == 1


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_duplicate_polygons(labeled_image):
    # test boundary labels are *not* duplicated
    mesh = labeled_image.contour_labels(
        duplicate_polygons=False,
        output_labels='boundary',
        output_mesh_type='quads',
    )
    actual_boundary_labels = mesh[BOUNDARY_LABELS]
    expected_boundary_labels = [
        [2.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [2.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 5.0],  # <-- Single internal boundary cell here
        [5.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
    ]
    assert np.array_equal(actual_boundary_labels, expected_boundary_labels)
    assert np.shape(expected_boundary_labels) == (14, 2)

    # test boundary labels *are* duplicated
    mesh = labeled_image.contour_labels(
        duplicate_polygons=True,
        output_labels='boundary',
        output_mesh_type='quads',
    )
    actual_boundary_labels = mesh[BOUNDARY_LABELS].tolist()
    expected_boundary_labels = [
        [2.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [2.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 5.0],  # <- Original cell, ascending order
        [5.0, 2.0],  # <- Test this cell is inserted, descending order
        [5.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
    ]
    assert np.array_equal(actual_boundary_labels, expected_boundary_labels)
    assert np.shape(expected_boundary_labels) == (15, 2)

    # Test surface labels requires duplicate_polygons=True
    match = (
        'Parameter duplicate_polygons must be True when generating surface labels with internal polygons.'
        '\nEither set duplicate_polygons to True or set internal_polygons to False.'
    )
    with pytest.raises(ValueError, match=match):
        labeled_image.contour_labels(duplicate_polygons=False, output_labels='surface')

    # Test surface labels equals first component of boundary labels
    mesh = labeled_image.contour_labels(
        duplicate_polygons=True,
        output_labels='surface',
        output_mesh_type='quads',
    )
    actual_surface_labels = mesh[SURFACE_LABELS]
    expected_surface_labels = np.array(expected_boundary_labels)[
        :,
        0,
    ]
    assert np.array_equal(actual_surface_labels, expected_surface_labels)
    assert np.shape(expected_surface_labels) == (15,)


@pytest.mark.parametrize(
    ('output_labels', 'array_names'),
    [
        ('boundary', [BOUNDARY_LABELS]),
        ('surface', [SURFACE_LABELS]),
        (True, [SURFACE_LABELS, BOUNDARY_LABELS]),
        (None, []),
    ],
)
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_output_labels(labeled_image, output_labels, array_names):
    assert labeled_image.array_names == ['labels']

    mesh = labeled_image.contour_labels(output_labels=output_labels)
    assert mesh.array_names == array_names

    if SURFACE_LABELS in mesh.array_names:
        assert mesh.active_scalars_name == SURFACE_LABELS


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_raises(labeled_image):
    match = 'Invalid output mesh type "invalid", use "quads" or "triangles"'
    with pytest.raises(ValueError, match=match):
        labeled_image.contour_labels(output_mesh_type='invalid')

    match = "Output labels must be one of 'surface', 'boundary', or True. Got all."
    with pytest.raises(ValueError, match=match):
        labeled_image.contour_labels(output_labels='all')

    # TODO: test float input vs int for select_input/output


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_invalid_scalars(labeled_image):
    # Nonexistent scalar key
    with pytest.raises(KeyError):
        labeled_image.contour_labels(scalars='nonexistent_key')

    # Using cell data
    labeled_image.cell_data['cell_data'] = np.zeros(labeled_image.n_cells)
    with pytest.raises(ValueError, match='Can only process point data'):
        labeled_image.contour_labels(scalars='cell_data')

    # When no scalars are given and active scalars are not point data
    labeled_image.set_active_scalars('cell_data', preference='cell')
    with pytest.raises(ValueError, match='active scalars must be point array'):
        labeled_image.contour_labels()


@pytest.fixture()
def single_point_image():
    image = pv.ImageData(dimensions=(1, 1, 1))
    image.point_data['image'] = 99
    image.point_data['other'] = 42
    return image


@pytest.mark.parametrize('pad_size', [1, 2])
@pytest.mark.parametrize('pad_value', [-1, 0, 1, 2])
@pytest.mark.parametrize('pad_singleton_dims', [True, False])
def test_pad_image(single_point_image, pad_size, pad_value, pad_singleton_dims):
    image_point_value = single_point_image['image'][0]
    if pad_singleton_dims:
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

    padded = single_point_image.pad_image(
        pad_size=pad_size,
        pad_value=pad_value,
        pad_singleton_dims=pad_singleton_dims,
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
    single_point_image,
    pad_size,
    expected_dimensions,
    expected_bounds,
):
    image_point_value = single_point_image['image'][0]
    pad_value = 7

    padded = single_point_image.pad_image(
        pad_size=pad_size,
        pad_singleton_dims=True,
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
    single_point_image,
    pad_size,
    expected_dimensions,
    expected_bounds,
):
    image_point_value = single_point_image['image'][0]
    other_point_value = single_point_image['other'][0]
    pad_value = 7

    padded = single_point_image.pad_image(
        pad_size=pad_size,
        pad_singleton_dims=True,
        pad_value=pad_value,
        pad_all_scalars=True,
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
    padded = single_point_image.pad_image(0, scalars=scalars, pad_all_scalars=all_scalars)
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
        pad_singleton_dims=True,
        pad_all_scalars=True,
        progress_bar=True,
    )
    assert np.array_equal(len(padded.active_scalars), np.prod(dims + pad_size * 2))
    assert np.all(padded.active_scalars == new_value)

    single_point_image['scalars2'] = [new_value * 2]

    padded = single_point_image.pad_image(
        'wrap',
        pad_size=pad_size,
        pad_singleton_dims=True,
        pad_all_scalars=True,
    )
    assert np.array_equal(len(padded.active_scalars), np.prod(dims + pad_size * 2))
    assert np.all(padded.active_scalars == new_value)
    assert np.all(padded['scalars2'] == new_value * 2)


def test_pad_image_raises(single_point_image, uniform, logo):
    match = 'Pad size cannot be negative. Got -1.'
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
        "\nTry setting `pad_all_scalars=False` or update the array."
    )
    logo.pad_image(pad_value=(0, 0, 0, 0), pad_all_scalars=False)
    with pytest.raises(ValueError, match=re.escape(match)):
        logo.pad_image(pad_value=(0, 0, 0, 0), pad_all_scalars=True)
