"""This module contains any tests which cause memory leaks."""
import weakref

import numpy as np


def test_pyvistandarray_assign(sphere):
    sphere.point_data['data'] = range(sphere.n_points)

    # this might leave a reference behind if we don't properly use the pointer
    # to the vtk array.
    sphere.point_data['data'] = sphere.point_data['data']


def test_pyvistandarray_strides(sphere):
    sphere['test_scalars'] = sphere.points[:, 2]
    assert np.allclose(sphere['test_scalars'], sphere.points[:, 2])


def test_complex_collection(plane):
    name = 'my_data'
    data = np.random.random((plane.n_points, 2)).view(np.complex128).ravel()
    plane.point_data[name] = data

    # ensure shallow copy
    assert np.shares_memory(plane.point_data[name], data)

    # ensure data remains but original numpy object does not
    ref = weakref.ref(data)
    data_copy = data.copy()
    del data
    assert np.allclose(plane.point_data[name], data_copy)

    assert ref() is None
