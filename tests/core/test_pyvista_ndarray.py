from __future__ import annotations

import re
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import vtk as _vtk

from pyvista import examples
from pyvista import pyvista_ndarray
from pyvista import vtk_points


@pytest.fixture
def pyvista_ndarray_1d():
    return pyvista_ndarray([1.0, 2.0, 3.0])


def test_slices_are_associated():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)

    # check that slices of pyvista_ndarray are associated correctly
    assert points[1, :].VTKObject == points.VTKObject
    assert points[1, :].dataset.Get() == points.dataset.Get()
    assert points[1, :].association == points.association


def test_copies_are_not_associated():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)
    points_2 = points.copy()

    # check that copies of pyvista_ndarray are dissociated from the original dataset
    assert points_2.VTKObject is None
    assert points_2.dataset is None
    assert points_2.association.name == 'NONE'
    assert not np.shares_memory(points, points_2)


def test_modifying_modifies_dataset():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)

    dataset_modified = mock.Mock()
    array_modified = mock.Mock()
    dataset.AddObserver(_vtk.vtkCommand.ModifiedEvent, dataset_modified)
    points.AddObserver(_vtk.vtkCommand.ModifiedEvent, array_modified)

    # __setitem__ calls dataset.Modified() and points.Modified()
    points[:] *= 0.5
    assert dataset_modified.call_count == 1
    assert array_modified.call_count == 1

    # __setitem__ with single-indices works does same
    points[0, 0] = 0.5
    assert dataset_modified.call_count == 2
    assert array_modified.call_count == 2

    # setting all new points calls dataset.Modified()
    dataset.points = points.copy()
    assert dataset_modified.call_count == 3
    assert array_modified.call_count == 2


# TODO: This currently doesn't work for single element indexing operations!
# in these cases, the __array_finalize__ method is not called
@pytest.mark.skip
def test_slices_are_associated_single_index():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)

    assert points[1, 1].VTKObject == points.VTKObject
    assert points[1, 1].dataset.Get() == points.dataset.Get()
    assert points[1, 1].association == points.association


def test_min(pyvista_ndarray_1d):
    arr = np.array(pyvista_ndarray_1d)
    assert pyvista_ndarray_1d.min() == arr.min()

    # also ensure that methods return float-like values just like numpy
    assert isinstance(pyvista_ndarray_1d.min(), type(arr.min()))


def test_squeeze(pyvista_ndarray_1d):
    reshaped_pvarr = pyvista_ndarray_1d.reshape((3, 1))
    assert np.array_equal(reshaped_pvarr.squeeze(), np.array(reshaped_pvarr.squeeze()))


def test_tobytes(pyvista_ndarray_1d):
    assert pyvista_ndarray_1d.tobytes() == np.array(pyvista_ndarray_1d).tobytes()


def test_add_1d():
    # ensure that 1d single value arrays match numpy
    pv_arr = pyvista_ndarray([1]) + pyvista_ndarray([1])
    np_arr = np.array([1]) + np.array([1])
    assert np.array_equal(pv_arr, np_arr)


@pytest.mark.parametrize('val', [1, True, None])
def test_raises(val):
    match = re.escape(
        f'pyvista_ndarray got an invalid type {type(val)}. '
        f'Expected an Iterable or vtk.vtkAbstractArray'
    )
    with pytest.raises(TypeError, match=match):
        pyvista_ndarray(val)


@pytest.mark.parametrize('obj_in', [np.eye(3), vtk_points(np.eye(3)).GetData()])
def test_wrap_pandas(obj_in):
    array = pyvista_ndarray(obj_in)
    df = pd.DataFrame(array)
    assert np.shares_memory(df.values, array)
