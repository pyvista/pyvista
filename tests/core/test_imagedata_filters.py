from dataclasses import dataclass
from typing import Iterable, Tuple

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


def FirstLabelInfo():
    """Define info for a labeled region for the labeled_image fixture."""

    @dataclass(frozen=True)
    class FirstLabel:
        label_id = 2

        # single point in center column,
        # surrounded by background,
        # adjacent to other label
        point_ids = (17,)

        # Cube sharing one side with other label
        n_quads_background_boundary = 5
        n_quads_internal_boundary = 1
        n_quads_total = n_quads_background_boundary + n_quads_internal_boundary

    return FirstLabel()


def SecondLabelInfo():
    """Define info for a labeled region for the labeled_image fixture."""

    @dataclass(frozen=True)
    class SecondLabel:
        label_id = 5

        # two points in center of column,
        # surrounded by background,
        # adjacent to other label,
        # one side touching image boundary
        point_ids = (18, 19)

        # 6 sides of cube * 2 points = 12 quad cells
        # minus 2 internal faces of this label that don't exist
        # minus 1 image boundary face
        # minus 1 shared face with other label
        n_quads_background_boundary = 8
        n_quads_internal_boundary = 1
        n_quads_total = n_quads_background_boundary + n_quads_internal_boundary

    return SecondLabel()


@pytest.fixture()
def labeled_image():
    # Create 4x3x3 image with two adjacent labels
    first_label_info = FirstLabelInfo()
    second_label_info = SecondLabelInfo()

    dim = (4, 3, 3)
    labels = np.zeros(np.prod(dim))
    labels[list(first_label_info.point_ids)] = first_label_info.label_id
    labels[list(second_label_info.point_ids)] = second_label_info.label_id
    image = pv.ImageData(dimensions=dim)
    image.point_data['labels'] = labels

    label_ids = np.unique(image.point_data.active_scalars).tolist()
    assert label_ids == [0, first_label_info.label_id, second_label_info.label_id]
    return image


def _get_contours(mesh, label_id: int):
    # Remove any cells which do not have the label_id
    is_background_boundary, is_internal_boundary = _classify_boundary_labels_cells(mesh)
    array = mesh[BOUNDARY_LABELS]
    is_label_boundary = np.any(array == label_id, axis=1)
    background_contour = mesh.extract_cells(
        np.logical_and(is_background_boundary, is_label_boundary),
    )
    internal_contour = mesh.extract_cells(
        np.logical_and(is_internal_boundary, is_label_boundary),
    )
    return background_contour, internal_contour


def _is_triangles(mesh):
    return all(cell.type == pv.CellType.TRIANGLE for cell in mesh.cell)


def _is_quads(mesh):
    return all(cell.type == pv.CellType.QUAD for cell in mesh.cell)


def _get_unique_ids(array, include_background_id=True):
    """Get unique region values from the array. Optionally include a background value of 0."""
    ids = np.unique(array).tolist()
    if not include_background_id:
        ids.remove(0) if 0 in ids else None
    return ids


def _get_unique_ids_background_boundary(mesh):
    """Return region ids from the boundary_labels array which share a boundary with the background."""
    background_boundaries, _ = _split_boundary_labels(mesh)
    return _get_unique_ids(background_boundaries, include_background_id=False)


def _get_unique_ids_internal_boundary(mesh):
    """Return region ids from the boundary_labels array which share internal boundaries."""
    _, internal_boundaries = _split_boundary_labels(mesh)
    return _get_unique_ids(internal_boundaries)


def _split_boundary_labels(mesh) -> Tuple[np.ndarray, np.ndarray]:
    # Split boundary labels array into two groups:
    #   foreground-foreground (internal) boundaries
    #   background-foreground boundaries
    array = mesh[BOUNDARY_LABELS]
    is_background_boundary, is_internal_boundary = _classify_boundary_labels_cells(mesh)
    background_boundaries = array[is_background_boundary]
    internal_boundaries = array[is_internal_boundary]
    return background_boundaries, internal_boundaries


def _classify_boundary_labels_cells(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Classify cells as boundary with background or internal boundary between foreground regions."""
    background_value = 0
    array = mesh[BOUNDARY_LABELS]
    is_background_boundary = np.any(array == background_value, axis=1)
    is_internal_boundary = np.invert(is_background_boundary)
    return is_background_boundary, is_internal_boundary


def _make_iterable(item):
    return [item] if not isinstance(item, Iterable) else item


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
    first_label_info = FirstLabelInfo()
    second_label_info = SecondLabelInfo()
    FIXED_PARAMS = dict(
        output_labels='boundary',
        independent_regions=False,
    )
    # Determine expected output
    if output_mesh_type == 'triangles' or output_mesh_type is None and smoothing:
        has_correct_celltype = _is_triangles
        multiplier = 2  # quads are subdivided into 2 triangles
    else:
        assert output_mesh_type == 'quads' or not smoothing
        has_correct_celltype = _is_quads
        multiplier = 1

    # Do test
    mesh = labeled_image.contour_labels(
        scalars=scalars,
        smoothing=smoothing,
        output_mesh_type=output_mesh_type,
        **FIXED_PARAMS,
    )
    assert BOUNDARY_LABELS in mesh.cell_data
    assert has_correct_celltype(mesh)
    label_ids = _get_unique_ids_background_boundary(mesh)
    assert label_ids == [first_label_info.label_id, second_label_info.label_id]

    info = first_label_info
    background_contour, internal_contour = _get_contours(mesh, info.label_id)
    expected_n_cells_background = info.n_quads_background_boundary * multiplier
    assert background_contour.n_cells == expected_n_cells_background
    expected_n_cells_internal = info.n_quads_internal_boundary * multiplier
    assert internal_contour.n_cells == expected_n_cells_internal

    if smoothing:
        assert mesh.area < 0.01
    else:
        assert mesh.area == mesh.n_cells / multiplier

    info = second_label_info
    background_contour, internal_contour = _get_contours(mesh, info.label_id)
    expected_n_cells_background = info.n_quads_background_boundary * multiplier
    assert background_contour.n_cells == expected_n_cells_background
    expected_n_cells_internal = info.n_quads_internal_boundary * multiplier
    assert internal_contour.n_cells == expected_n_cells_internal


select_cases = [
    None,
    FirstLabelInfo().label_id,
    SecondLabelInfo().label_id,
    [FirstLabelInfo().label_id, SecondLabelInfo().label_id],
]


@pytest.mark.parametrize('select_inputs', select_cases)
@pytest.mark.parametrize(
    'select_outputs',
    select_cases,
)
@pytest.mark.parametrize('internal_boundaries', [True, False])
@pytest.mark.parametrize('image_boundaries', [True, False])
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_boundaries(
    labeled_image,
    select_inputs,
    select_outputs,
    internal_boundaries,
    image_boundaries,
):
    # Test correct boundary_labels output
    # Fix params boundary_labels=True and independent_regions=False for tests since we want
    # to test the unmodified 'stock' boundary_labels array output from vtkSurfaceNets
    FIXED_PARAMS = dict(output_labels='boundary', independent_regions=False)
    ALL_LABEL_IDS = _get_unique_ids(labeled_image['labels'], include_background_id=False)

    mesh = labeled_image.contour_labels(
        select_inputs=select_inputs,
        select_outputs=select_outputs,
        internal_boundaries=internal_boundaries,
        **FIXED_PARAMS,
    )  # , image_boundary_faces=image_boundary_faces)
    assert BOUNDARY_LABELS in mesh.cell_data
    actual_output_ids = _get_unique_ids_background_boundary(mesh)

    # Make sure param values are iterable
    select_inputs_iter = _make_iterable(select_inputs if select_inputs else ALL_LABEL_IDS)
    select_outputs_iter = _make_iterable(select_outputs if select_outputs else ALL_LABEL_IDS)

    # All selected outputs are expected if it's also selected at the input
    expected_output_ids = [id_ for id_ in select_outputs_iter if id_ in select_inputs_iter]

    assert actual_output_ids == expected_output_ids

    if internal_boundaries and len(select_inputs_iter) == 2:
        # The two labels share a boundary by default
        # Boundary exists if no labels removed from input
        expected_shared_region_ids = ALL_LABEL_IDS
        shared_cells = _split_boundary_labels(mesh)[1]
        assert len(shared_cells) == 2
    else:
        expected_shared_region_ids = []

    actual_shared_region_ids = _get_unique_ids_internal_boundary(mesh)
    assert actual_shared_region_ids == expected_shared_region_ids

    # Make sure temp array created for select_inputs is removed
    assert labeled_image.array_names == ['labels']


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

    # When no scalas are given and active scalars are not point data
    labeled_image.set_active_scalars('cell_data', preference='cell')
    with pytest.raises(ValueError, match='active scalars must be point array'):
        labeled_image.contour_labels()
