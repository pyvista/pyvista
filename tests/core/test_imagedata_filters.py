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
    with pytest.raises(ValueError):
        label_map.contour_labeled(output_mesh_type='invalid')


@pytest.mark.skipif(not VTK93, reason='At least VTK 9.3 is required')
def test_contour_labeled_with_invalid_output_style():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    with pytest.raises(NotImplementedError):
        label_map.contour_labeled(output_style='selected')

    with pytest.raises(ValueError):
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
