"""This module contains any tests which cause memory leaks."""
import numpy as np


def test_pyvistandarray_assign(sphere):
    sphere.point_data['data'] = range(sphere.n_points)

    # this might leave a reference behind if we don't property use the pointer
    # to the vtk array.
    sphere.point_data['data'] = sphere.point_data['data']


def test_pyvistandarray_strides(sphere):
    sphere['test_scalars'] = sphere.points[:, 2]
    assert np.allclose(sphere['test_scalars'], sphere.points[:, 2])
