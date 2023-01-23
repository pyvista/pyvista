"""Test unstructured grid filters."""
import pytest


@pytest.mark.needs_vtk_version(9, 0, 3)
def test_clean(hexbeam):
    hexbeam_b = hexbeam.copy()
    hexbeam_b.points[:, 0] += 1
    merged = hexbeam.merge(hexbeam_b, merge_points=False)
    merged.n_points

    cleaned = merged.clean()
    assert cleaned.n_cells == merged.n_cells
    assert cleaned.n_points < merged.n_points
