"""Test dataset filters.

New tests for data_set should be placed here rather in ../test_filters.py

"""
import numpy as np

import pyvista as pv


def test_merge_points():
    """Test DataSetFilters.merge_points."""
    cyl = pv.CylinderStructured(
        radius=np.linspace(0, 0.01, 5),
        height=0.5,
        center=(0.0, 0.0, 0.25),
        direction=(0.0, 0.0, 1.0),
    )
    grid = cyl.merge_points()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_points < cyl.n_points
