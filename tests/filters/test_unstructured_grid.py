"""Test unstructured grid filters."""
import numpy as np
import pytest


@pytest.mark.needs_vtk_version(9, 0, 3)
def test_clean(hexbeam):
    hexbeam_b = hexbeam.copy()
    hexbeam_b.points[:, 0] += 1
    merged = hexbeam.merge(hexbeam_b, merge_points=False)
    merged.n_points
    merged['comp_6_pt'] = np.random.random((merged.n_points, 6))
    merged['comp_6_cell'] = np.random.random((merged.n_cells, 6))

    cleaned = merged.clean()
    assert cleaned.n_cells == merged.n_cells
    assert cleaned.n_points < merged.n_points

    assert 'comp_6_pt' in cleaned.point_data
    assert 'comp_6_cell' in cleaned.cell_data
    assert 'OriginalPointIds' in cleaned.point_data

    # ids = cleaned.point_data['OriginalPointIds']
    # breakpoint()
    # assert np.allclose(cleaned.points, merged.points[ids])
    # diff = np.linalg.norm(cleaned.points- merged.points[ids],axis=1) > 0
    # cleaned.plot(scalars=diff, off_screen=False)
