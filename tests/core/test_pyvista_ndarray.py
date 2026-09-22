from __future__ import annotations

import gc
import re

import numpy as np
import pandas as pd
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista import examples
from pyvista import pyvista_ndarray
from pyvista import vtk_points
from pyvista.core.utilities.arrays import FieldAssociation


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


def test_dataset_from_attributes_wrapper():
    sphere = pv.Sphere()
    attributes = sphere.point_data
    array = pv.pyvista_ndarray(attributes.VTKObject.GetArray('Normals'), dataset=attributes)
    assert array.dataset.Get() is attributes.VTKObject


def test_copies_are_not_associated():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)
    points_2 = points.copy()

    # check that copies of pyvista_ndarray are dissociated from the original dataset
    assert points_2.VTKObject is None
    assert points_2.dataset is None
    assert points_2.association.name == 'NONE'
    assert not np.shares_memory(points, points_2)


class _CallCounter:
    """Count observer calls, without recording who made them.

    Not ``unittest.mock.Mock``: a mock keeps every call's arguments, one of which is
    the VTK object the observer is attached to. That cycle runs through a VTK wrapper,
    which the collector can neither traverse nor clear, so the object it observes is
    leaked for the life of the process. The same cycle between two ordinary Python
    objects is collected.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, *args) -> None:  # noqa: ARG002
        self.call_count += 1


def test_modifying_modifies_dataset():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)

    dataset_modified = _CallCounter()
    array_modified = _CallCounter()
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


@pytest.mark.parametrize('squeeze', [pyvista_ndarray.squeeze, np.squeeze])
@pytest.mark.parametrize(
    ('shape', 'axis'),
    [
        ((1,), None),
        ((1, 1), None),
        ((1, 1), (0, 1)),
        ((1, 3, 1), -1),
        ((1, 3, 1), (0, -1)),
        ((3,), None),
        ((1,), ()),
        ((), None),
        ((), 0),
        ((), -1),
        ((0, 1), None),
    ],
)
def test_squeeze(squeeze, shape, axis):
    """Squeeze follows NumPy shape rules and returns an array view."""
    array = pyvista_ndarray(np.arange(np.prod(shape, dtype=int)).reshape(shape))
    expected = np.asarray(array).squeeze(axis)
    result = squeeze(array, axis)
    assert isinstance(result, pyvista_ndarray)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert np.array_equal(result, expected)
    if result.shape == array.shape:
        assert result is array
    if array.size:
        assert np.shares_memory(result, array)
        result[...] = 42
        assert np.all(np.asarray(array) == 42)


@pytest.mark.parametrize('squeeze', [pyvista_ndarray.squeeze, np.squeeze])
@pytest.mark.parametrize('axis', [1, 3, -4, (0, 0), (0, -3), (0, 1), 1.5, [0]])
def test_squeeze_invalid_axis(squeeze, axis):
    """Invalid axes raise the same exception as NumPy."""
    array = pyvista_ndarray(np.ones((1, 3, 1)))
    with pytest.raises((ValueError, IndexError, TypeError)) as error:
        np.asarray(array).squeeze(axis=axis)
    with pytest.raises(type(error.value)):
        squeeze(array, axis=axis)


@pytest.mark.parametrize('squeeze', [pyvista_ndarray.squeeze, np.squeeze])
@pytest.mark.parametrize('step', [2, -1])
def test_squeeze_strided(squeeze, step):
    """Squeezing non-contiguous and reversed views retains their storage."""
    array = pyvista_ndarray(np.arange(24).reshape(4, 1, 6)).transpose(2, 1, 0)[::step]
    result = squeeze(array)
    assert isinstance(result, pyvista_ndarray)
    assert np.array_equal(result, np.asarray(array).squeeze())
    assert np.shares_memory(result, array)
    result[0, 0] = 42
    assert array[0, 0, 0] == 42


@pytest.mark.parametrize('squeeze', [pyvista_ndarray.squeeze, np.squeeze])
@pytest.mark.parametrize(
    ('dtype', 'value'),
    [
        (np.int64, 2),
        (np.float64, 2.5),
        (np.bool_, False),
        (np.complex64, 2 + 3j),
        (np.complex128, 2 + 3j),
    ],
)
def test_squeeze_associated(squeeze, dtype, value):
    """Singleton views retain metadata and notify their dataset on writes."""
    mesh = pv.PolyData(np.zeros((1, 3)))
    mesh.point_data['values'] = np.ones(1, dtype=dtype)
    array = mesh.point_data['values']
    assert isinstance(array, pyvista_ndarray)
    result = squeeze(array.reshape((1, 1)))
    assert isinstance(result, pyvista_ndarray)
    assert result.shape == ()
    assert result.dtype == dtype
    assert result.item() == 1
    assert np.shares_memory(result, array)
    assert result.dataset is array.dataset
    assert result.dataset.Get() is mesh
    assert result.VTKObject is array.VTKObject
    assert result.association == array.association == FieldAssociation.POINT
    dataset_modified, array_modified = _CallCounter(), _CallCounter()
    mesh.AddObserver(_vtk.vtkCommand.ModifiedEvent, dataset_modified)
    array.AddObserver(_vtk.vtkCommand.ModifiedEvent, array_modified)
    result[...] = value
    assert mesh.point_data['values'].item() == value
    assert dataset_modified.call_count == array_modified.call_count == 1


@pytest.mark.parametrize('reduction', ['sum', 'min', 'max'])
@pytest.mark.parametrize('size', [1, 3])
def test_reductions_return_scalars(reduction, size):
    """Reductions retain NumPy scalar return types."""
    array = pyvista_ndarray(np.arange(size))
    expected = getattr(np.asarray(array), reduction)()
    result = getattr(array, reduction)()
    assert type(result) is type(expected)
    assert result == expected


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


def test_no_dataset_does_not_allocate_weak_reference():
    # Regression test for https://github.com/pyvista/pyvista/issues/8532
    arr = pyvista_ndarray([1.0, 2.0, 3.0])
    assert arr.dataset is None


def _count_vtk_weak_references() -> int:
    """Count live ``vtkWeakReference`` objects, tolerating dead weak refs.

    ``isinstance`` on some weakref-like proxies whose targets have already
    been collected raises ``ReferenceError``; skip those.
    """
    count = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, _vtk.vtkWeakReference):
                count += 1
        except ReferenceError:
            continue
    return count


def test_point_data_assignment_does_not_leak_vtk_weak_reference():
    # Regression test for https://github.com/pyvista/pyvista/issues/8532
    # Compare counts before and after rather than asserting an absolute
    # zero — other code in the interpreter (other tests, fixtures,
    # imported plugins) may legitimately hold ``vtkWeakReference``
    # instances that have nothing to do with this operation.
    gc.collect()
    before = _count_vtk_weak_references()

    mesh = pv.Sphere()
    mesh.point_data['data'] = mesh.points[:, 2].astype(float)
    del mesh
    gc.collect()

    after = _count_vtk_weak_references()
    assert after <= before, (
        f'point_data assignment leaked {after - before} vtkWeakReference instance(s)'
    )


def test_unassociated_array_stores_no_metadata():
    arr = pyvista_ndarray([1.0, 2.0, 3.0])
    assert arr.VTKObject is None
    assert arr.dataset is None
    assert arr.association == FieldAssociation.NONE
    assert not {'VTKObject', 'dataset', 'association'} & set(vars(arr))


def test_association_without_dataset_propagates_to_views():
    arr = pyvista_ndarray([1.0, 2.0, 3.0], association=FieldAssociation.POINT)
    assert arr[1:].association == FieldAssociation.POINT
    assert (arr + 1).association == FieldAssociation.NONE


def test_copies_and_ufunc_results_are_not_associated():
    points = pv.Sphere().points
    assert points.VTKObject is not None
    for result in (
        points + 1,
        points[[0, 1]],
        points[points[:, 2] > 0],
        points.astype(np.float64),
        np.abs(points),
    ):
        assert isinstance(result, pyvista_ndarray)
        assert result.VTKObject is None
        assert result.dataset is None
        assert result.association == FieldAssociation.NONE
    assert points.T.VTKObject is points.VTKObject
    assert points.reshape(-1).dataset is points.dataset


def test_dataset_reference_targets_owner():
    mesh = pv.Sphere()
    assert mesh.points.dataset.Get() is mesh
    assert mesh.point_data['Normals'].dataset.Get() is mesh
    assert mesh.point_data.active_normals.dataset.Get() is mesh


def test_detached_array_attribute_error_names_the_attribute():
    match = re.escape("'pyvista_ndarray' object has no attribute 'GetNumberOfTuples'")
    with pytest.raises(AttributeError, match=match):
        pyvista_ndarray([1.0, 2.0]).GetNumberOfTuples()
