""" test vtkInterface.utilities """
import pytest
import numpy as np
from vtkInterface import utilities


def test_createvectorpolydata_error():
    orig = np.random.random((3, 1))
    vec = np.random.random((3, 1))
    with pytest.raises(Exception):
        utilities.CreateVectorPolyData(orig, vec)


def test_createvectorpolydata_1D():
    orig = np.random.random(3)
    vec = np.random.random(3)
    vdata = utilities.CreateVectorPolyData(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.GetPointScalars('vectors'))


def test_createvectorpolydata():
    orig = np.random.random((100, 3))
    vec = np.random.random((100, 3))
    # with pytest.raises(Exception):
    vdata = utilities.CreateVectorPolyData(orig, vec)
    assert np.any(vdata.points)
    assert np.any(vdata.GetPointScalars('vectors'))
