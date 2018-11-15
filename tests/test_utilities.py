""" test vtki.utilities """
import pytest
import numpy as np
from vtki import utilities


def test_createvectorpolydata_error():
    orig = np.random.random((3, 1))
    vec = np.random.random((3, 1))
    with pytest.raises(Exception):
        utilities.vector_poly_data(orig, vec)


def test_createvectorpolydata_1D():
    orig = np.random.random(3)
    vec = np.random.random(3)
    vdata = utilities.vector_poly_data(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.point_arrays['vectors'])


def test_createvectorpolydata():
    orig = np.random.random((100, 3))
    vec = np.random.random((100, 3))
    vdata = utilities.vector_poly_data(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.point_arrays['vectors'])
