from unittest import mock

import numpy as np
import pytest
import vtk as _vtk

from pyvista import examples, pyvista_ndarray


@pytest.fixture
def pv_ndarray_1d():
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


def test_min(pv_ndarray_1d):
    arr = np.array(pv_ndarray_1d)
    assert pv_ndarray_1d.min() == arr.min()

    # also ensure that methods return float-like values just like numpy
    assert isinstance(pv_ndarray_1d.min(), type(arr.min()))


def test_squeeze(pv_ndarray_1d):
    reshaped_pvarr = pv_ndarray_1d.reshape((3, 1))
    assert np.array_equal(reshaped_pvarr.squeeze(), np.array(reshaped_pvarr.squeeze()))


def test_tobytes(pv_ndarray_1d):
    assert pv_ndarray_1d.tobytes() == np.array(pv_ndarray_1d).tobytes()


def test_add_1d():
    # ensure that 1d single value arrays match numpy
    pv_arr = pyvista_ndarray([1]) + pyvista_ndarray([1])
    np_arr = np.array([1]) + np.array([1])
    assert np.array_equal(pv_arr, np_arr)
