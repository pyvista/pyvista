from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pytest

import pyvista as pv
from pyvista import DeprecationError, examples


@pytest.fixture()
def frog_tissue():
    return examples.download_frog_tissue()


deprecation_match = 'This filter produces unexpected results and is deprecated.'


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = frog_tissue
        # Extract surface for each label
        mesh = label_map.contour_labeled()

        assert label_map.point_data.active_scalars.max() == 29
        assert 'BoundaryLabels' in mesh.cell_data
        assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled_with_smoothing(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = frog_tissue

        # Extract smooth surface for each label
        mesh = label_map.contour_labeled(smoothing=True)
        # this somehow mutates the object... also the n_labels is likely not correct

        assert 'BoundaryLabels' in mesh.cell_data
        assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled_with_reduced_labels_count(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = frog_tissue

        # Extract surface for each label
        mesh = label_map.contour_labeled(n_labels=2)
        # this somehow mutates the object... also the n_labels is likely not correct

        assert 'BoundaryLabels' in mesh.cell_data
        assert np.max(mesh['BoundaryLabels'][:, 0]) == 2


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled_with_triangle_output_mesh(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = frog_tissue

        # Extract surface for each label
        mesh = label_map.contour_labeled(scalars='MetaImage', output_mesh_type='triangles')

        assert 'BoundaryLabels' in mesh.cell_data
        assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled_with_boundary_output_style(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = frog_tissue

        # Extract surface for each label
        mesh = label_map.contour_labeled(output_style='boundary')

        assert 'BoundaryLabels' in mesh.cell_data
        assert np.max(mesh['BoundaryLabels'][:, 0]) == 29


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled_with_invalid_output_mesh_type(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = frog_tissue

        # Extract surface for each label
        with pytest.raises(ValueError):  # noqa: PT011
            label_map.contour_labeled(output_mesh_type='invalid')


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled_with_invalid_output_style(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = frog_tissue

        # Extract surface for each label
        with pytest.raises(NotImplementedError):
            label_map.contour_labeled(output_style='selected')

        with pytest.raises(ValueError):  # noqa: PT011
            label_map.contour_labeled(output_style='invalid')


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled_with_scalars(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        # and create a new array with reduced number of labels
        label_map = frog_tissue
        label_map['labels'] = label_map['MetaImage'] // 2

        # Extract surface for each label
        mesh = label_map.contour_labeled(scalars='labels')

        assert 'BoundaryLabels' in mesh.cell_data
        assert np.max(mesh['BoundaryLabels'][:, 0]) == 14


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labeled_with_invalid_scalars(frog_tissue):
    with pytest.raises(DeprecationError, match=deprecation_match):  # noqa: PT012
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = frog_tissue

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


def first_label_info():
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


def second_label_info():
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
def labeled_image(first_label_info=first_label_info(), second_label_info=second_label_info()):
    # Create 4x3x3 image with two adjacent labels

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
    is_background_boundary, is_foreground_boundary = _classify_BoundaryLabels_cells(mesh)
    array = mesh['BoundaryLabels']
    is_label_boundary = np.any(array == label_id, axis=1)
    background_contour = mesh.extract_cells(
        np.logical_and(is_background_boundary, is_label_boundary),
    )
    foreground_contour = mesh.extract_cells(
        np.logical_and(is_foreground_boundary, is_label_boundary),
    )
    return background_contour, foreground_contour


def _is_triangles(mesh):
    return all(cell.type == pv.CellType.TRIANGLE for cell in mesh.cell)


def _is_quads(mesh):
    return all(cell.type == pv.CellType.QUAD for cell in mesh.cell)


def _get_unique_ids_BoundaryLabels_with_background(mesh):
    BoundaryLabels_background_boundary, _ = _split_BoundaryLabels(mesh)
    return _get_unique_ids(BoundaryLabels_background_boundary, include_background_id=False)


def _get_unique_ids(array, include_background_id=True):
    ids = np.unique(array).tolist()
    if not include_background_id:
        ids.remove(0) if 0 in ids else None
    return ids


def _get_unique_ids_BoundaryLabels_with_foreground(mesh):
    _, BoundaryLabels_foreground_boundary = _split_BoundaryLabels(mesh)
    return _get_unique_ids(BoundaryLabels_foreground_boundary)


def _split_BoundaryLabels(mesh) -> Tuple[np.ndarray, np.ndarray]:
    # Split boundary labels array into two groups:
    #   foreground-foreground boundaries
    #   background-foreground boundaries
    array = mesh['BoundaryLabels']
    is_background_boundary, is_foreground_boundary = _classify_BoundaryLabels_cells(mesh)
    BoundaryLabels_background_boundary = array[is_background_boundary]
    BoundaryLabels_foreground_boundary = array[is_foreground_boundary]
    return BoundaryLabels_background_boundary, BoundaryLabels_foreground_boundary


def _classify_BoundaryLabels_cells(mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Classify cells as boundary with background region or between two foreground regions."""
    background_value = 0
    array = mesh['BoundaryLabels']
    is_background_boundary = np.any(array == background_value, axis=1)
    is_foreground_boundary = np.invert(is_background_boundary)
    return is_background_boundary, is_foreground_boundary


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
    first_label_info=first_label_info(),
    second_label_info=second_label_info(),
):
    FIXED_PARAMS = dict(
        boundary_labels=True,
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
    assert 'BoundaryLabels' in mesh.cell_data
    assert has_correct_celltype(mesh)
    label_ids = _get_unique_ids_BoundaryLabels_with_background(mesh)
    assert label_ids == [first_label_info.label_id, second_label_info.label_id]

    info = first_label_info
    background_contour, foreground_contour = _get_contours(mesh, info.label_id)
    expected_n_cells_background = info.n_quads_background_boundary * multiplier
    assert background_contour.n_cells == expected_n_cells_background
    expected_n_cells_foreground = info.n_quads_internal_boundary * multiplier
    assert foreground_contour.n_cells == expected_n_cells_foreground

    if smoothing:
        assert mesh.area < 0.01
    else:
        assert mesh.area == mesh.n_cells / multiplier

    info = second_label_info
    background_contour, foreground_contour = _get_contours(mesh, info.label_id)
    expected_n_cells_background = info.n_quads_background_boundary * multiplier
    assert background_contour.n_cells == expected_n_cells_background
    expected_n_cells_foreground = info.n_quads_internal_boundary * multiplier
    assert foreground_contour.n_cells == expected_n_cells_foreground


selected_cases = [
    None,
    first_label_info().label_id,
    second_label_info().label_id,
    [first_label_info().label_id, second_label_info().label_id],
]


@pytest.mark.parametrize('select_inputs', selected_cases)
@pytest.mark.parametrize(
    'select_outputs',
    selected_cases,
)
@pytest.mark.parametrize('internal_boundaries', [True, False])
@pytest.mark.parametrize('image_boundaries', [True, False])
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_output_surface(
    labeled_image,
    select_inputs,
    select_outputs,
    internal_boundaries,
    image_boundaries,
):
    FIXED_PARAMS = dict(boundary_labels=True, independent_regions=False)
    ALL_LABEL_IDS = _get_unique_ids(labeled_image['labels'], include_background_id=False)

    mesh = labeled_image.contour_labels(
        select_inputs=select_inputs,
        select_outputs=select_outputs,
        internal_boundaries=internal_boundaries,
        **FIXED_PARAMS,
    )  # , image_boundary_faces=image_boundary_faces)
    assert 'BoundaryLabels' in mesh.cell_data
    actual_output_ids = _get_unique_ids_BoundaryLabels_with_background(mesh)

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
        shared_cells = _split_BoundaryLabels(mesh)[1]
        assert len(shared_cells) == 2
    else:
        expected_shared_region_ids = []

    actual_shared_region_ids = _get_unique_ids_BoundaryLabels_with_foreground(mesh)
    assert actual_shared_region_ids == expected_shared_region_ids


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_independent_regions_surface_labels(labeled_image):
    # Disable smoothing so output has simple quad geometry
    # Include boundary labels for testing
    FIXED_PARAMS = dict(smoothing=False, boundary_labels=True)

    # Test internal boundary is shared
    mesh = labeled_image.contour_labels(independent_regions=False, **FIXED_PARAMS)
    assert mesh.active_scalars_name == 'SurfaceLabels'

    actual_shared_boundary_labels = mesh['BoundaryLabels'].tolist()
    expected_shared_boundary_labels = [
        [2.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [2.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 5.0],  # <-- Single shared boundary cell here
        [5.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
        [5.0, 0.0],
    ]
    assert actual_shared_boundary_labels == expected_shared_boundary_labels
    assert np.shape(expected_shared_boundary_labels) == (14, 2)

    # Test surface labels is norm of boundary labels
    actual_shared_surface_labels = mesh['SurfaceLabels'].tolist()
    expected_shared_surface_labels = np.linalg.norm(
        expected_shared_boundary_labels,
        axis=1,
    ).tolist()
    assert actual_shared_surface_labels == expected_shared_surface_labels
    assert np.shape(expected_shared_surface_labels) == (14,)

    # Test internal boundary is *not* shared
    mesh = labeled_image.contour_labels(independent_regions=True, **FIXED_PARAMS)
    assert mesh.active_scalars_name == 'SurfaceLabels'

    actual_independent_boundary_labels = mesh['BoundaryLabels'].tolist()
    expected_independent_boundary_labels = [
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
    assert actual_independent_boundary_labels == expected_independent_boundary_labels
    assert np.shape(expected_independent_boundary_labels) == (15, 2)

    # Test surface labels equals first component of boundary labels
    actual_independent_surface_labels = mesh['SurfaceLabels'].tolist()
    expected_independent_surface_labels = np.array(expected_independent_boundary_labels)[
        :,
        0,
    ].tolist()
    assert actual_independent_surface_labels == expected_independent_surface_labels
    assert np.shape(expected_independent_surface_labels) == (15,)


@pytest.mark.parametrize('surface_labels', [True, False])
@pytest.mark.parametrize('boundary_labels', [True, False])
@pytest.mark.needs_vtk_version(9, 3, 0)
def test_contour_labels_output_arrays(labeled_image, surface_labels, boundary_labels):
    SURFACE_LABELS = 'SurfaceLabels'
    BOUNDARY_LABELS = 'BoundaryLabels'
    assert labeled_image.array_names == ['labels']

    mesh = labeled_image.contour_labels()
    assert mesh.array_names == [SURFACE_LABELS]

    mesh = labeled_image.contour_labels(
        surface_labels=surface_labels,
        boundary_labels=boundary_labels,
    )
    if surface_labels:
        assert mesh.active_scalars_name == SURFACE_LABELS
        assert SURFACE_LABELS in mesh.cell_data

    if boundary_labels:
        assert BOUNDARY_LABELS in mesh.cell_data

    if surface_labels and boundary_labels:
        assert mesh.array_names == [SURFACE_LABELS, BOUNDARY_LABELS]
    if not surface_labels and not boundary_labels:
        assert mesh.array_names == []

    assert labeled_image.array_names == ['labels']


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
