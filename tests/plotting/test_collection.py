"""This module contains any tests which cause memory leaks."""

import numpy as np


def test_pyvistandarray_assign(sphere):
    sphere.clear_data()
    arr = np.ones((sphere.n_points, 3))
    sphere.point_data['data'] = arr

    # this might leave a reference behind if we don't property use the pointer
    # to the vtk array.
    sphere.point_data['data'] = sphere.point_data['data']


def test_pyvistandarray_strides(sphere):
    sphere['test_scalars'] = sphere.points[:, 2]
    assert np.allclose(sphere['test_scalars'], sphere.points[:, 2])
