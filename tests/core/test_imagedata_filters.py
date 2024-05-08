import numpy as np
import pytest

import pyvista as pv
from pyvista import DeprecationError

SURFACE_LABELS = 'surface_labels'
BOUNDARY_LABELS = 'boundary_labels'


def test_contour_labeled_deprecated():
    match = 'This filter produces unexpected results and is deprecated.'
    with pytest.raises(DeprecationError, match=match):
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
    background_boundaries = (
        extracted[BOUNDARY_LABELS] if BOUNDARY_LABELS in extracted.cell_data else []
    )
    return _get_label_ids(background_boundaries)


def _get_ids_with_internal_boundary(mesh):
    """Return region ids from the boundary_labels array which share internal boundaries."""
    extracted = mesh.extract_values(0, invert=True, component_mode=1)
    internal_boundaries = (
        extracted[BOUNDARY_LABELS] if BOUNDARY_LABELS in extracted.cell_data else []
    )
    return _get_label_ids(internal_boundaries)


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
        independent_regions=False,
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
@pytest.mark.parametrize('internal_boundaries', [True, False])
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_boundaries(
    labeled_image,
    select_inputs,
    select_outputs,
    internal_boundaries,
):
    # Test correct boundary_labels output
    ALL_LABEL_IDS = np.array([2, 5])

    mesh = labeled_image.contour_labels(
        select_inputs=select_inputs,
        select_outputs=select_outputs,
        internal_boundaries=internal_boundaries,
        output_labels='boundary',
        independent_regions=False,
    )  # , image_boundary_faces=image_boundary_faces)
    assert BOUNDARY_LABELS in mesh.cell_data
    actual_output_ids = _get_ids_with_background_boundary(mesh)

    # Make sure param values are iterable
    select_inputs_iter = np.atleast_1d(select_inputs) if select_inputs else ALL_LABEL_IDS
    select_outputs_iter = np.atleast_1d(select_outputs) if select_outputs else ALL_LABEL_IDS

    # All selected outputs are expected if it's also selected at the input
    expected_output_ids = [id_ for id_ in select_outputs_iter if id_ in select_inputs_iter]

    assert actual_output_ids == expected_output_ids

    if internal_boundaries and len(select_inputs_iter) == 2:
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
    mesh_true = labeled_image.contour_labels(image_boundaries=True, output_mesh_type='quads')
    mesh_false = labeled_image.contour_labels(image_boundaries=False, output_mesh_type='quads')
    assert mesh_true.n_cells - mesh_false.n_cells == 1


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_independent_regions(labeled_image):
    # test boundary labels are *not* independent
    mesh = labeled_image.contour_labels(
        independent_regions=False,
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

    # test boundary labels *are* independent
    mesh = labeled_image.contour_labels(
        independent_regions=True,
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

    # Test surface labels requires independent_regions=True
    match = (
        'Parameter independent_regions cannot be False when generating surface labels with internal boundaries.'
        '\nEither set independent_regions to True or set internal_boundaries to False.'
    )
    with pytest.raises(ValueError, match=match):
        labeled_image.contour_labels(independent_regions=False, output_labels='surface')

    # Test surface labels equals first component of boundary labels
    mesh = labeled_image.contour_labels(
        independent_regions=True,
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
        ('all', [SURFACE_LABELS, BOUNDARY_LABELS]),
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
