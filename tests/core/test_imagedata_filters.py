import numpy as np

import pyvista as pv
from pyvista import examples


def test_contour_labeled(sphere):
    if pv.vtk_version_info >= (9, 3, 0):
        # Load a 3D label map (segmentation of a frog's tissue)
        label_map = examples.download_frog_tissue()

        # Extract surface for each label
        mesh = label_map.contour_labeled()

        assert label_map.point_data.active_scalars.max() == 29
        assert 'BoundaryLabels' in mesh.cell_data
        assert np.max(mesh['BoundaryLabels']) == 29
