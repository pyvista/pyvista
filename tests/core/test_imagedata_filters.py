import numpy as np
import pytest

import pyvista as pv
from pyvista import examples

VTK93 = pv.vtk_version_info >= (9, 3)


@pytest.mark.skipif(not VTK93, reason="At least VTK 9.3 is required")
def test_contour_labeled():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    mesh = label_map.contour_labeled()

    assert label_map.point_data.active_scalars.max() == 29
    assert "BoundaryLabels" in mesh.cell_data
    assert np.max(mesh["BoundaryLabels"][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason="At least VTK 9.3 is required")
def test_contour_labeled_with_smoothing():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract smooth surface for each label
    mesh = label_map.contour_labeled(smoothing=True)
    # this somehow mutates the object... also the n_labels is likely not correct

    assert "BoundaryLabels" in mesh.cell_data
    assert np.max(mesh["BoundaryLabels"][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason="At least VTK 9.3 is required")
def test_contour_labeled_with_reduced_labels_count():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    mesh = label_map.contour_labeled(n_labels=2)
    # this somehow mutates the object... also the n_labels is likely not correct

    assert "BoundaryLabels" in mesh.cell_data
    assert np.max(mesh["BoundaryLabels"][:, 0]) == 2


@pytest.mark.skipif(not VTK93, reason="At least VTK 9.3 is required")
def test_contour_labeled_with_triangle_output_mesh():
    # Load a 3D label map (segmentation of a frog's tissue)
    label_map = examples.download_frog_tissue()

    # Extract surface for each label
    mesh = label_map.contour_labeled(scalars="MetaImage")

    assert "BoundaryLabels" in mesh.cell_data
    assert np.max(mesh["BoundaryLabels"][:, 0]) == 29


@pytest.mark.skipif(not VTK93, reason="At least VTK 9.3 is required")
def test_contour_labeled_with_scalars():
    # Load a 3D label map (segmentation of a frog's tissue)
    # and create a new array with reduced number of labels
    label_map = examples.download_frog_tissue()
    label_map["labels"] = label_map["MetaImage"] // 2

    # Extract surface for each label
    mesh = label_map.contour_labeled(scalars="labels")

    assert "BoundaryLabels" in mesh.cell_data
    assert np.max(mesh["BoundaryLabels"][:, 0]) == 14
